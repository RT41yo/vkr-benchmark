"""Stable experiment inputs: prompts and secret streams."""

from vkr_benchmark.inputs.prompts import PromptRecord, PromptRegistry
from vkr_benchmark.inputs.secrets import create_secret_source

__all__ = ["PromptRecord", "PromptRegistry", "create_secret_source"]
