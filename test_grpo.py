# test_grpo.py
import os
import sys
import argparse
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
import sentencepiece as spm
import grain.python as grain

# Make sure project root is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.finetune.TextToSQLGRPOTrainer import TextToSQLGRPOTrainer
from src.data_loading.load_sources import get_sources
from src.data_loading.text_utils import evaluate_text_to_sql
from src.model_loading.llm_factory import LLMFactoryConfig, LLMModules, LLMModuleFactory
from tunix.models.gemma3 import model as gemma_lib
from tunix.generate import sampler as sampler_lib
from flax import nnx
import orbax.checkpoint as ocp


class PromptMappingLoader:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __iter__(self):
        for batch in self.data_loader:
            if 'prompts' in batch and 'prompt' not in batch:
                batch = dict(batch)
                batch['prompt'] = batch['prompts']
            yield batch

    def __len__(self):
        return len(self.data_loader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Gemma 3 Text-to-SQL Model on the Test Set.")

    # Base paths
    parser.add_argument("--ckpt_path", type=str, default="./models/gemma-3-270m-it/1", help="Path to base weights.")
    parser.add_argument("--tokenizer_path", type=str, default="./models/tokenizer.model", help="Path to tokenizer.")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Where LoRA checkpoints are saved.")
    
    # Evaluation options
    parser.add_argument("--checkpoint_step", type=int, default=None, help="LoRA step checkpoint to load. Defaults to latest.")
    parser.add_argument("--no_lora", action="store_true", help="Evaluate the base model without LoRA adapters.")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of test samples to evaluate. -1 for all.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum generated tokens per SQL query.")
    parser.add_argument("--max_token_len", type=int, default=1055, help="Filter prompts to this max token length.")

    args = parser.parse_args()
    args.save_dir = os.path.abspath(args.save_dir)
    args.ckpt_path = os.path.abspath(args.ckpt_path)

    print("=" * 80)
    print("TEXT-TO-SQL GRPO MODEL EVALUATION")
    print("=" * 80)
    print(f"Base Checkpoint Path:   {args.ckpt_path}")
    print(f"Tokenizer Path:         {args.tokenizer_path}")
    print(f"LoRA Checkpoint Dir:    {args.save_dir}")
    print(f"No LoRA (Base Model):   {args.no_lora}")
    print(f"Evaluation Samples:     {args.num_samples}")
    print(f"Max New Tokens:         {args.max_new_tokens}")
    print("=" * 80)

    # 1. Load Data
    print("Loading test data source...")
    train, dev, test = get_sources()
    
    # Load tokenizer for prompt length filtering if needed
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer_path)

    if args.max_token_len is not None:
        print(f"Filtering test source to max token length of {args.max_token_len}...")
        test.filter_by_token_length(tokenizer, args.max_token_len)

    # Create test loader with batch_size=1 for sequential evaluation
    test_loader = grain.load(
        test,
        num_epochs=1,
        shuffle=False,
        batch_size=1,
        worker_count=0,
    )
    test_loader = PromptMappingLoader(test_loader)
    print(f"Test set loaded with {len(test)} records.")

    # 2. Configure the LLM Factory Mesh
    device_grid = mesh_utils.create_device_mesh((len(jax.devices()), 1))
    mesh = Mesh(device_grid, axis_names=("fsdp", "tp"))

    config = LLMFactoryConfig(
        ckpt_path=args.ckpt_path,
        model_config=gemma_lib.ModelConfig.gemma3_270m(),
        mesh=mesh,
        tokenizer_path=args.tokenizer_path,
        include_model=True,
        include_tokenizer=True,
        include_sampler=True,
    )

    # 3. Initialize Trainer and Load checkpoint if necessary
    trainer = TextToSQLGRPOTrainer(
        lora_rank=8,  # Match training lora rank
        checkpoint_dir=args.save_dir,
    )

    # 4. Set up the correct sampler based on model choice
    if args.no_lora:
        print("\nEvaluating base model (without LoRA)...")
        base_modules = LLMModuleFactory.build(config)
        eval_model = base_modules.model
        tokenizer = base_modules.tokenizer
    else:
        trainer.initialize_model(config, load_step=args.checkpoint_step)
        print("\nEvaluating fine-tuned model with LoRA adapter...")
        eval_model = trainer.actor_model
        tokenizer = trainer.modules.tokenizer
        
        # Log info about the checkpoints
        all_steps = trainer.ckpt_manager.all_steps()
        latest = trainer.ckpt_manager.latest_step()
        print(f"Available checkpoints: {list(all_steps)}")
        print(f"Using checkpoint step:  {args.checkpoint_step if args.checkpoint_step is not None else latest}")

    cache_config = sampler_lib.CacheConfig(
        cache_size=config.max_prompt_length + config.max_generation_steps + 256,
        num_layers=config.model_config.num_layers,
        num_kv_heads=config.model_config.num_kv_heads,
        head_dim=config.model_config.head_dim,
    )

    eval_sampler = sampler_lib.Sampler(
        transformer=eval_model,
        tokenizer=tokenizer,
        cache_config=cache_config,
    )

    eval_modules = LLMModules(
        model=eval_model,
        tokenizer=tokenizer,
        sampler=eval_sampler,
    )

    # 5. Run Evaluation
    actual_num_samples = args.num_samples if args.num_samples != -1 else len(test)
    print(f"\nStarting evaluation of {actual_num_samples} samples from the test set...")
    eval_results = evaluate_text_to_sql(
        modules=eval_modules,
        data_loader=test_loader,
        num_samples=actual_num_samples,
        max_new_tokens=args.max_new_tokens,
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print(f"Accuracy: {eval_results['accuracy'] * 100:.2f}% ({eval_results['correct']}/{eval_results['total']})")
    print("=" * 80)


if __name__ == "__main__":
    main()
