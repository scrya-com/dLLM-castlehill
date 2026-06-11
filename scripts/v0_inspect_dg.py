#!/usr/bin/env python3
"""v0 sanity check: Load DiffusionGemma, inspect, run forward pass."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion

MODEL_PATH = "models/diffusiongemma-26B-A4B-it"
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.cuda.empty_cache()

    print("GPU:", torch.cuda.get_device_name(0))

    # ── 1. Load ──────────────────────────────────────────────
    print("=== Load ===")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"  vocab={tok.vocab_size} bos={tok.bos_token_id} eos={tok.eos_token_id} mask={tok.mask_token_id}")

    # Load BF16 on CPU — no quantization trickery for v0 inspection
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.cpu().eval()
    dev = model.device
    print(f"  loaded BF16 on {dev}, canvas_length={model.config.canvas_length}")

    # ── 2. Architecture ──────────────────────────────────────
    print("\n=== Architecture ===")
    tc = model.config.text_config
    enc = model.model.encoder
    dec = model.model.decoder

    print(f"  layers={tc.num_hidden_layers} hidden={tc.hidden_size} head_dim={tc.head_dim}")
    print(f"  experts={tc.num_experts} top_k={tc.top_k_experts} moe_intermediate={tc.moe_intermediate_size}")
    print(f"  total_params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    print(f"  embed_scale={dec.embed_tokens.embed_scale}")
    print(f"  logit_softcap={model.final_logit_softcapping}")
    print(f"  sc_mlp_params={sum(p.numel() for p in dec.self_conditioning.parameters())/1e6:.1f}M")
    print(f"  layer_types={list(set(tc.layer_types))}")

    # Per-layer scalars (encoder layers in .language_model.layers)
    enc_layers = enc.language_model.layers
    dec_layers = dec.layers
    for i in [0, 14, 29]:
        es = enc_layers[i].layer_scalar
        ds = dec_layers[i].layer_scalar
        print(f"  layer[{i}] enc_scalar={es.item():.4f} dec_scalar={ds.item():.4f}")

    # ── 3. Encoder → Decoder forward ─────────────────────────
    print("\n=== Forward pass ===")
    prompt = "Explain quantum computing in simple terms."
    p_ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    c_len = model.config.canvas_length
    c_ids = torch.randint(0, tok.vocab_size, (1, c_len)).to(dev)

    with torch.no_grad():
        # Use inputs_embeds path to avoid device_map meta-tensor bug
        p_emb = model.get_input_embeddings()(p_ids)
        enc_out = enc(inputs_embeds=p_emb)
        c_emb = dec.embed_tokens(c_ids)
        dec_out = dec(decoder_input_ids=c_ids, past_key_values=enc_out.past_key_values,
                       self_conditioning_logits=None)
        logits = model.lm_head(dec_out.last_hidden_state)
        logits = logits / model.final_logit_softcapping
        logits = torch.tanh(logits) * model.final_logit_softcapping

    print(f"  logits: {logits.shape} range=[{logits.min():.2f}, {logits.max():.2f}]")
    print(f"  top-1: {logits[0, :5].argmax(-1).tolist()}")

    # ── 4. KV cache analysis ─────────────────────────────────
    print("\n=== KV cache ===")
    with torch.no_grad():
        p_emb = model.get_input_embeddings()(p_ids)
        enc_out = enc(inputs_embeds=p_emb)
    kv_len = enc_out.past_key_values.get_seq_length()
    print(f"  encoder prefill → KV cache seq_len={kv_len}")

    with torch.no_grad():
        dec_out = dec(decoder_input_ids=c_ids, past_key_values=enc_out.past_key_values,
                       self_conditioning_logits=None)
    print(f"  decoder output: {dec_out.last_hidden_state.shape}")

    # ── 5. VFM compatibility ─────────────────────────────────
    print("\n=== VFM compatibility ===")
    D = tc.hidden_size

    # Test get_input_embeddings
    if hasattr(model, 'get_input_embeddings'):
        emb = model.get_input_embeddings()
        print(f"  [✓] get_input_embeddings() → {type(emb).__name__}")
    else:
        print(f"  [✗] get_input_embeddings() MISSING")

    # Test encoder with inputs_embeds
    test_e = torch.randn(1, 64, D).to(model.device).bfloat16()
    try:
        with torch.no_grad():
            eo = enc(inputs_embeds=test_e)
        print(f"  [✓] encoder(inputs_embeds=...) → {eo.last_hidden_state.shape}")
    except Exception as e:
        print(f"  [✗] encoder(inputs_embeds): {e}")

    # Test decoder — does NOT accept inputs_embeds (we must monkey-patch)
    try:
        with torch.no_grad():
            dec(decoder_input_ids=c_ids[:, :16], past_key_values=enc_out.past_key_values,
                inputs_embeds=test_e[:, :16, :])
        print(f"  [✓] decoder(inputs_embeds=...) works via kwargs")
    except Exception as e:
        print(f"  [✗] decoder refuses inputs_embeds — needs monkey-patch ({str(e)[:80]})")

    # Test output_hidden_states
    try:
        with torch.no_grad():
            eo = enc(inputs_embeds=test_e, output_hidden_states=True)
        print(f"  [✓] encoder output_hidden_states → {len(eo.hidden_states)} layers")
    except Exception as e:
        print(f"  [✗] output_hidden_states: {e}")

    # ── 6. Embedding sphere radius ───────────────────────────
    print("\n=== Embedding sphere ===")
    w = dec.embed_tokens.weight.float()
    norms = w.norm(dim=-1)
    print(f"  embed_weight_norm: mean={norms.mean():.4f} std={norms.std():.4f} min={norms.min():.4f} max={norms.max():.4f}")
    print(f"  embed_norm (for VFMv5): {norms.mean():.4f}")

    # Test what happens when canvas embeds go through SC MLP
    print("\n=== Self-conditioning path ===")
    c_emb = dec.embed_tokens(c_ids)
    print(f"  canvas_emb norm: {c_emb.norm(dim=-1).mean():.4f}")
    sc_emb = dec.self_conditioning(c_emb, torch.zeros_like(c_emb))
    print(f"  after SC(zero_cond): {sc_emb.shape}, norm={sc_emb.norm(dim=-1).mean():.4f}")

    print("\n✓ v0 complete")


if __name__ == "__main__":
    main()
