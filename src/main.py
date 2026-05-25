import argparse
import os
import sys
from gemma import gm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_loading.load_sources import get_sources, create_dataloaders
from src.data_loading.text_utils import evaluate_text_to_sql
from src.model_loading.llm_factory import LLMFactoryConfig, LLMModuleFactory

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Gemma model on text-to-sql tasks.")
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="./models/gemma-3-270m", 
        help="Path to the model checkpoint directory."
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default="./models/tokenizer.model", 
        help="Path to the tokenizer model file."
    )
    args = parser.parse_args()

    print(f"Checking if checkpoint exists: {os.path.exists(args.ckpt_path)}")
    print(f"Checking if tokenizer exists: {os.path.exists(args.tokenizer_path)}")
    
    config = LLMFactoryConfig(
        ckpt_path=args.ckpt_path,
        model_class=gm.nn.Gemma3_270M,
        tokenizer_class=gm.text.Gemma3Tokenizer,
        tokenizer_path=args.tokenizer_path,
        include_model=True,
        include_tokenizer=True,
        include_sampler=True,
    )

    modules = LLMModuleFactory.build(config)

    train, dev, test = get_sources()
    train_loader, dev_loader, test_loader = create_dataloaders(train, dev, test)
    print('type of ', type(modules.model))
    # Usage:
    eval_results = evaluate_text_to_sql(
        modules=modules,
        data_loader=dev_loader,
        num_samples=10,
        max_new_tokens=100,
        temperature=0.7,
        verbose=True,
    )

    # Access individual results
    for result in eval_results['results']:
        if not result['is_correct']:
            print(f"Failed on: {result['question']}")
            print(f"  Expected: {result['ground_truth']}")
            print(f"  Got: {result['predicted_sql']}")


if __name__ == "__main__":
    main()
