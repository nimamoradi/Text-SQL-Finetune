# src/model_loading/llm_factory.py

import logging
from dataclasses import dataclass
from typing import Any, Optional

from gemma import gm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMFactoryConfig:
    """
    Simple config for Gemma JAX models.

    ckpt_path: Path to checkpoint directory (local or GCS)
    model_class: Pass the actual model class (e.g., gm.nn.Gemma2_270M)
    tokenizer_class: Pass the tokenizer class (e.g., gm.text.Gemma2Tokenizer)
    tokenizer_path: Optional path to tokenizer model file
    """
    ckpt_path: str
    model_class: Any
    tokenizer_class: Any
    tokenizer_path: Optional[str] = None

    include_model: bool = True
    include_tokenizer: bool = False
    include_sampler: bool = True


@dataclass
class LLMModules:
    """Container for loaded modules"""
    model: Optional[Any] = None
    params: Optional[Any] = None
    tokenizer: Optional[Any] = None
    sampler: Optional[Any] = None


class LLMModuleFactory:

    @staticmethod
    def build(config: LLMFactoryConfig) -> LLMModules:
        """Build only what you asked for"""
        modules = LLMModules()

        if config.include_model:
            logger.info("Loading model from: %s", config.ckpt_path)

            # Instantiate model architecture
            model = config.model_class()

            # Load checkpoint
            ckpt_path = config.ckpt_path

            params = gm.ckpts.load_params(ckpt_path)

            modules.model = model
            modules.params = params
            logger.info("Model loaded ✓")

        if config.include_tokenizer:
            logger.info("Loading tokenizer")

            if config.tokenizer_path:
                modules.tokenizer = config.tokenizer_class(config.tokenizer_path)
            else:
                modules.tokenizer = config.tokenizer_class()
            logger.info("Tokenizer loaded ✓")

        if config.include_sampler:
            if modules.model is None or modules.params is None:
                raise ValueError("include_sampler=True requires include_model=True")
            
            logger.info("Attaching sampler")
            modules.sampler = gm.text.Sampler(
                model=modules.model,
                params=modules.params,
            )
            logger.info("Sampler attached ✓")

        return modules
