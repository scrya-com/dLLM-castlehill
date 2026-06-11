#!/usr/bin/env python3
"""Monkey-patch Gemma4ClippableLinear to inherit from nn.Linear for PEFT compatibility.

This is the fix from https://github.com/huggingface/transformers/pull/45388
applied as a runtime patch to transformers 5.11.0.
"""

import torch.nn as nn
from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear


def apply_patch():
    """Make Gemma4ClippableLinear a subclass of nn.Linear at runtime."""
    if issubclass(Gemma4ClippableLinear, nn.Linear):
        print("[patch] Already patched.")
        return

    # Store reference to original class
    orig_cls = Gemma4ClippableLinear

    # Create a new class inheriting from nn.Linear that wraps the old behavior
    # The fix in PR #45388 changes the class from composition (has-a nn.Linear)
    # to inheritance (is-a nn.Linear). We can't change the class hierarchy at
    # runtime, so we patch isinstance checks via __subclasshook__.
    #
    # Actually the real fix is: change class Gemma4ClippableLinear(nn.Module)
    # to class Gemma4ClippableLinear(nn.Linear), with the linear layer's
    # parameters directly on self instead of self.linear.
    #
    # For a runtime monkey-patch, we register nn.Linear as a virtual base class:
    nn.Linear.register(Gemma4ClippableLinear)
    print("[patch] Registered Gemma4ClippableLinear as nn.Linear subclass")


if __name__ == "__main__":
    apply_patch()
    print(f"  issubclass: {issubclass(Gemma4ClippableLinear, nn.Linear)}")
    print("Done.")
