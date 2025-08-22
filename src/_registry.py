"""
Registry file for Inspect AI extensions.

This module imports all custom providers and extensions to ensure they are
registered with Inspect AI when the package is loaded.
"""

# Register HF prefill provider
from inspect_ai.model._registry import modelapi  # type: ignore

@modelapi(name="hf_prefill")
def hf_prefill():
    from .hf_prefill import HFPrefillAPI

    return HFPrefillAPI
