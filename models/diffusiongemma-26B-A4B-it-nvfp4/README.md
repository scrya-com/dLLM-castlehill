
---
pipeline_tag: text-generation
base_model: google/diffusiongemma-26B-A4B-it
license: apache-2.0
license_name: apache-license-2.0
license_link: https://ai.google.dev/gemma/apache_2
tags:
- nvidia
- ModelOpt
- DiffusionGemma-26B-A4B-IT
- quantized
- NVFP4
- nvfp4
---

# Model Overview

## Description:
DiffusionGemma 26B A4B IT is an open-weights multimodal generative model developed by Google DeepMind that processes text, image, and video inputs to produce text output via discrete diffusion. Built on the Gemma 4 26B A4B Mixture-of-Experts (MoE) architecture with 25.2B total parameters and 3.8B active parameters, the model employs an encoder-decoder design with bidirectional attention that generates tokens in parallel 256-token blocks, enabling high-speed generation exceeding 1,100 tokens per second at low batch sizes on NVIDIA Hopper H100 (FP8). DiffusionGemma 26B A4B IT supports a 256K token context window, configurable thinking (reasoning) mode, native function calling, and multilingual inference across 35+ languages.
The NVIDIA DiffusionGemma 26B A4B IT NVFP4 model is quantized with [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer).


This model is ready for commercial and non-commercial use. <br>

## Third-Party Community Consideration
This model is not owned or developed by NVIDIA. This model has been developed and built to a third-party's requirements for this application and use case; see link to Non-NVIDIA [DiffusionGemma 26B A4B Model Card](https://huggingface.co/google/diffusiongemma-26B-A4B-it).

## License/Terms of Use:
 SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 SPDX-License-Identifier: Apache-2.0

Governing Terms: [Apache 2.0](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4/blob/main/apache_2.0_license.md) [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).


### Deployment Geography:
Global

### Use Case:
**Use Case:** DiffusionGemma 26B A4B IT is designed for developers, researchers, and enterprises requiring high-speed multimodal text generation. Supported use cases include conversational AI and chatbots, text summarization, code generation and step-by-step reasoning, image and document understanding (OCR, chart comprehension, PDF parsing, screen and UI parsing), video content analysis, agentic workflows with native function calling, and multilingual NLP tasks across 35+ languages.

### Release Date:
Hugging Face 06/10/2026 via https://huggingface.co/nvidia/DiffusionGemma-26B-A4B-IT-NVFP4

## Model Architecture:
**Architecture Type:** Transformer  
**Network Architecture:** Mixture-of-Experts  
**Total Parameters:** 25.2B  
**Active Parameters:** 3.8B  
**Vocabulary Size:** 262,144  
**Base Model:** Gemma 4 26B A4B

### Input:
**Input Types:** Text, Image, Video  
**Input Formats:** String, Red, Green, Blue (RGB),  Video (MP4/WebM)
**Input Parameters:** One-Dimensional (1D), Two-Dimensional (2D), Three-Dimensional (3D)  
**Other Input Properties:** Images support variable aspect ratios and resolutions via a configurable visual token budget (70, 140, 280, 560, or 1120 tokens per image); place image content before text for optimal multimodal performance. Videos are processed as frame sequences up to 60 seconds at 1 frame per second.  
**Input Context Length (ISL):** 262,144

### Output:
**Output Types:** Text  
**Output Format:** String  
**Output Parameters:** One-Dimensional (1D)  
**Other Output Properties:** Text is generated in parallel 256-token canvas blocks via diffusion sampling with adaptive stopping; supports native function calling and structured JSON output formatting.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:
**Runtime Engine(s):**
* vLLM

**Supported Hardware Microarchitecture Compatibility:**
* NVIDIA Blackwell
* NVIDIA Hopper(H100)

**Preferred Operating System(s):**
* Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):
This model is NVFP4 quantized with nvidia-modelopt

## Training and Evaluation Datasets:

We calibrated the model using the dataset noted below, and performed evaluation using the benchmarks noted under Evaluation Datasets.
We did not perform training or testing for this Model Optimizer release. The methods noted under Training and Testing Datasets below represent the data collection and labeling methods used by the third-party to train and test the underlying DiffusionGemma 26B A4B model.<br>

## Calibration Dataset:
**Link:** [cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail), [Nemotron-Post-Training-Dataset-v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2)  
**Data Collection Method by dataset:** Automated.  
**Labeling Method by dataset:** Automated.  
**Properties:** The `cnn_dailymail` dataset contains English-language news articles and summaries. `Nemotron-Post-Training-Dataset-v2` is a post-training dataset curated by NVIDIA containing multi-turn conversations across diverse topics.


## Training Dataset:
**Data Modality:** Text, Image  
**Text Training Data Size:** Undisclosed  
**Image Training Data Size:** Undisclosed  
**Training Data Collection:** Hybrid: Automated, Human  
**Training Labeling:** Hybrid: Automated, Human  
**Training Properties:** Large-scale, diverse pre-training corpus spanning web documents, code, mathematics, and images across 140+ languages with a data collection cutoff of January 2025. Preprocessing includes CSAM filtering, sensitive personal information removal, and content quality filtering aligned with Google's AI policies.

## Evaluation Dataset:
**Datasets:** GPQA Diamond, AIME 2025, MMLU Pro, GSM8K, HumanEval, MMLU-0 Shot, IFEval 
**Data Collection Method by dataset:** Hybrid, Automated, Human  
**Labeling Method by dataset:** Hybrid, Automated, Human  
**Properties:** GPQA Diamond contains 448 graduate-level multiple-choice questions written by domain experts in biology, physics, and chemistry; AIME 2025 contains problems from the American Invitational Mathematics Examination; MMLU Pro is a multi-task language understanding benchmark with challenging multiple-choice questions across diverse academic domains

## Inference:
**Engine:** vLLM

**Test Hardware:** NVIDIA Blackwell B100

## Post Training Quantization
This model was obtained by quantizing the weights and activations of Gemma-26B-A4B-IT to NVFP4 data type, ready for inference with vLLM. This optimization reduces the number of bits per parameter from 16 to 4, reducing disk size and GPU memory requirements significantly.

## Usage
To serve this checkpoint with [vLLM](https://hub.docker.com/layers/vllm/vllm-openai/gemma-x86_64-cu130), you can launch the docker image `vllm/vllm-openai:gemma` and run the sample command below:

```
VLLM_USE_V2_MODEL_RUNNER=1
vllm serve nvidia/diffusiongemma-26B-A4B-IT-NVFP4 
 --trust-remote-code 
 --max-num-seqs 4 
 --attention-backend TRITON_ATTN  
 --enable-auto-tool-choice 
 --tool-call-parser gemma4 
 --reasoning-parser gemma4 
 --override-generation-config '{"max_new_tokens": null}' 
 --default-chat-template-kwargs '{"enable_thinking":true}'
```
Disclaimer: The vllm serve commands below are tentative and subject to change until the supporting vLLM image is publicly released. To check whether the image is live, see the vLLM releases page (https://github.com/vllm-project/vllm/releases) or the vllm/vllm-openai:gemma4 tags on Docker Hub. 

### Evaluation

The accuracy benchmark results (with thinking enabled) compared to the BF16 baseline are presented in the table below:

| Benchmark | Baseline (Full Precision) | NVFP4 |
|---|---|---|
| GPQA Diamond | 69.4% | 68.6% |
| AIME 2025 | 68.33% | 67.33% |
| GSM8K | 94.54% | 94.01% |
| IFEval | 94.01% | 94.56% |
| HumanEval | 94.09% | 95.00% |
| MMMLU 0 Shot | 88.50% | 88.13% |
| MMLU Pro | 81.0% | 80.7% |

> Benchmarking parameters: default temperature, top_p settings as in [orginal MC](https://huggingface.co/google/diffusiongemma-26B-A4B-it)

## Model Limitations:
The base model was trained on data that contains toxic language and societal biases originally crawled from the internet. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts. The model may generate answers that may be inaccurate, omit key information, or include irrelevant or redundant text producing socially unacceptable or undesirable text, even if the prompt itself does not include anything explicitly offensive.

## Ethical Considerations
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/)
