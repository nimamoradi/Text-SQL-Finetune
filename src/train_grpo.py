# src/train_grpo.py
import os
import argparse
# pyrefly: ignore [missing-import]
import jax
import warnings

# Suppress the massive MPS buffer donation warnings that spam the terminal
warnings.filterwarnings("ignore", message="Some donated buffers were not usable")

# --- Imports ---
from tunix.models.gemma3 import model as gemma_lib
from jax.experimental import mesh_utils
from jax.sharding import Mesh
import sentencepiece as spm

from src.finetune.TextToSQLGRPOTrainer import TextToSQLGRPOTrainer
from src.data_loading.load_sources import get_sources, create_dataloaders
from src.data_loading.text_utils import evaluate_text_to_sql, sql_correctness_reward
from src.model_loading.llm_factory import LLMFactoryConfig, LLMModuleFactory


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 3 using Tunix GRPO for Text-to-SQL.")

    # Base paths
    parser.add_argument("--ckpt_path", type=str, default="./models/gemma-3-270m", help="Path to base weights.")
    parser.add_argument("--tokenizer_path", type=str, default="./models/tokenizer.model", help="Path to tokenizer.")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Where to save LoRA checkpoints.")

    # RL/Training Hyperparameters
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Peak learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per step.")
    parser.add_argument("--generations", type=int, default=3, help="Number of rollouts per prompt (G).")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank of the LoRA adapter.")
    parser.add_argument("--max_token_len", type=int, default=1055, help="Filter prompts to this max token length.")
    parser.add_argument("--beta", type=float, default=0.04, help="KL divergence penalty coefficient to prevent mode collapse.")

    args = parser.parse_args()
    args.save_dir = os.path.abspath(args.save_dir)
    args.ckpt_path = os.path.abspath(args.ckpt_path)

    print(f"Checking if base checkpoint exists: {os.path.exists(args.ckpt_path)}")
    print(f"Checking if tokenizer exists: {os.path.exists(args.tokenizer_path)}")

    # 1. Load Data
    # Assuming get_sources returns your JsonDataSource and create_dataloaders wraps them in Grain
    print(f"Loading sources and filtering to {args.max_token_len} tokens...")
    train, dev, test = get_sources()
    print("Loading tokenizer for data filtering...")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer_path)
    train_loader, dev_loader, test_loader = create_dataloaders(
        train, dev, test, 
        tokenizer=tokenizer, 
        max_token_len=args.max_token_len
    )

    # 3. Configure the LLM Factory
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

    # 4. Initialize the GRPO Trainer
    trainer = TextToSQLGRPOTrainer(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_generations=args.generations,
        lora_rank=args.lora_rank,
        beta=args.beta,
        checkpoint_dir=args.save_dir,
    )

    # Wire up the components
    trainer.initialize_model(config)
    trainer.data_loader = train_loader  # Assign the filtered training data
    trainer.add_reward_function(sql_correctness_reward)

    # 5. Run Training
    trainer.train(steps=args.steps)

    # 6. Post-Training Evaluation
    print("\nTraining complete. Running evaluation on Dev set...")
    eval_results = evaluate_text_to_sql(
        modules=trainer.modules,
        data_loader=dev_loader,
        num_samples=100, # Evaluate more samples
        max_new_tokens=256,
        verbose=True,
    )

    # Print failures for analysis
    for result in eval_results['results']:
        if not result['is_correct']:
            print(f" Failed DB: {result.get('db_id', 'Unknown')}")
            print(f"  Expected: {result['ground_truth']}")
            print(f"  Got:      {result['predicted_sql']}")


if __name__ == "__main__":
    main()