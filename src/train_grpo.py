# src/scripts/train_grpo.py
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Force JAX to treat model weights as 16-bit bfloat by default
import jax
jax.config.update("jax_default_matmul_precision", "bfloat16")

import argparse
import os
import sys

from tunix.models.gemma3 import model as gemma_lib

from jax.experimental import mesh_utils
from jax.sharding import Mesh
from src.finetune.TextToSQLGRPOTrainer import TextToSQLGRPOTrainer

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import your existing data and factory modules
from src.data_loading.load_sources import get_sources, create_dataloaders
from src.data_loading.text_utils import evaluate_text_to_sql, sql_correctness_reward
from src.model_loading.llm_factory import LLMFactoryConfig, LLMModuleFactory


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 3 using Tunix GRPO for Text-to-SQL.")

    # Base paths
    parser.add_argument("--ckpt_path", type=str, default="./models/gemma-3-270m", help="Path to base weights.")
    parser.add_argument("--tokenizer_path", type=str, default="./models/tokenizer.model", help="Path to tokenizer.")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Where to save LoRA checkpoints.")

    # RL/Training Hyperparameters
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Peak learning rate.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per step.")
    parser.add_argument("--generations", type=int, default=4, help="Number of rollouts per prompt (G).")
    parser.add_argument("--lora_rank", type=int, default=16, help="Rank of the LoRA adapter.")

    args = parser.parse_args()

    print(f"Checking if base checkpoint exists: {os.path.exists(args.ckpt_path)}")
    print(f"Checking if tokenizer exists: {os.path.exists(args.tokenizer_path)}")

    # 1. Load Data
    # Assuming get_sources returns your JsonDataSource and create_dataloaders wraps them in Grain
    train, dev, test = get_sources()
    train_loader, dev_loader, test_loader = create_dataloaders(train, dev, test, 5000)

    # 2. Configure the LLM Factory (Loads pure Google weights)
    # config = LLMFactoryConfig(
    #     ckpt_path=args.ckpt_path,
    #     model_class=gm.nn.Gemma3_270M,
    #     tokenizer_class=gm.text.Gemma3Tokenizer,
    #     tokenizer_path=args.tokenizer_path,
    #     include_model=True,
    #     include_tokenizer=True,
    #     include_sampler=True,  # Needed for mid-training evaluation
    # )

    device_grid = mesh_utils.create_device_mesh((len(jax.devices()), 1))
    mesh = Mesh(device_grid, axis_names=("fsdp", "tp"))

    config = LLMFactoryConfig(
        ckpt_path=args.ckpt_path,
        model_config=gemma_lib.ModelConfig.gemma3_270m(),  # <-- This replaces model_class
        mesh=mesh,  # <-- Pass the mesh here
        tokenizer_path=args.tokenizer_path,
        include_model=True,
        include_tokenizer=True,
        include_sampler=True,
    )

    # 3. Initialize the GRPO Trainer
    trainer = TextToSQLGRPOTrainer(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_generations=args.generations,
        lora_rank=args.lora_rank,
        checkpoint_dir=args.save_dir
    )

    # Wire up the components
    trainer.initialize_model(config)  # Builds model via factory and applies LoRA
    trainer.data_loader = train_loader  # Bind your existing Grain dataloader
    trainer.add_reward_function(sql_correctness_reward)  # Inject your pure python validation logic

    # 4. Run the Training Loop
    trainer.train(steps=args.steps)

    # 5. Post-Training Evaluation
    # Because the trainer updates trainer.modules.params under the hood,
    # passing it to your existing eval function will test the newly fine-tuned LoRA weights.
    print("\nTraining complete. Running immediate evaluation on Dev set...")

    eval_results = evaluate_text_to_sql(
        modules=trainer.modules,
        data_loader=dev_loader,
        num_samples=10,
        max_new_tokens=100,
        verbose=True,
    )

    for result in eval_results['results']:
        if not result['is_correct']:
            # Make sure these keys match the exact dictionary returned by evaluate_text_to_sql
            print(f" Failed DB: {result.get('db_id', 'Unknown')}")
            print(f"  Expected: {result['ground_truth']}")
            print(f"  Got:      {result['predicted_sql']}")


if __name__ == "__main__":
    main()