# src/model_loading/llm_factory.py

import logging
from dataclasses import dataclass
from typing import Any, Optional

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from flax import nnx

from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LLMFactoryConfig:
    """
    Config for Tunix/Flax NNX Gemma 3 models.
    """
    ckpt_path: str
    model_config: Any  # e.g., gemma_lib.ModelConfig.gemma3_1b_it()
    mesh: Mesh         # Required by NNX for sharding state
    tokenizer_path: Optional[str] = None

    # Sampler generation limits (needed for KV cache pre-allocation)
    max_prompt_length: int = 256
    max_generation_steps: int = 768

    include_model: bool = True
    include_tokenizer: bool = False
    include_sampler: bool = True

    @staticmethod
    def create_default_mesh() -> Mesh:
        """
        Creates the standard 2D ('fsdp', 'tp') mesh required by Tunix Gemma 3.
        Allocates all available devices to FSDP, and 1 to TP.
        """
        num_devices = len(jax.devices())
        device_grid = mesh_utils.create_device_mesh((num_devices, 1))
        return Mesh(device_grid, axis_names=("fsdp", "tp"))


@dataclass
class LLMModules:
    """Container for loaded NNX modules"""
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    sampler: Optional[Any] = None


class LLMModuleFactory:

    @staticmethod
    def build(config: LLMFactoryConfig) -> LLMModules:
        """Build the NNX model and inference sampler."""
        modules = LLMModules()

        if config.include_model:
            logger.info("Loading NNX model from: %s", config.ckpt_path)

            # In NNX, we load the weights and instantiate the architecture simultaneously
            # within the JAX sharding mesh context.
            with config.mesh:
                base_model = params_safetensors_lib.create_model_from_safe_tensors(
                    config.ckpt_path, config.model_config, config.mesh
                )

            modules.model = base_model
            logger.info("NNX Model loaded ✓")

        if config.include_tokenizer:
            logger.info("Loading tokenizer")
            if config.tokenizer_path:
                modules.tokenizer = tokenizer_adapter.Tokenizer(tokenizer_path=config.tokenizer_path)
            else:
                raise ValueError("tokenizer_path is required for the Tunix tokenizer")
            logger.info("Tokenizer loaded ✓")

        if config.include_sampler:
            if modules.model is None:
                raise ValueError("include_sampler=True requires include_model=True")
            if modules.tokenizer is None:
                raise ValueError("include_sampler=True requires include_tokenizer=True for Tunix")

            logger.info("Attaching NNX sampler for inference")

            # The Tunix sampler requires a pre-configured KV Cache
            cache_config = sampler_lib.CacheConfig(
                cache_size=config.max_prompt_length + config.max_generation_steps + 256,
                num_layers=config.model_config.num_layers,
                num_kv_heads=config.model_config.num_kv_heads,
                head_dim=config.model_config.head_dim,
            )

            modules.sampler = sampler_lib.Sampler(
                transformer=modules.model,
                tokenizer=modules.tokenizer,
                cache_config=cache_config,
            )
            logger.info("NNX Sampler attached ✓")

        return modules