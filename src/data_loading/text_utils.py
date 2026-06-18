import re
from typing import Dict, Any
import sql_compare
import sqlparse


def is_valid_syntax(sql: str) -> bool:
    if not sql:
        return False
    try:
        parsed = sqlparse.parse(sql)
        return len(parsed) > 0 and parsed[0].get_type() != 'UNKNOWN'
    except Exception:
        return False

def extract_sql_from_response(text: str) -> str:
    """Extract SQL query from model output"""
    if not text:
        return ""
        
    text = str(text)

    # Try to find SQL after common markers
    patterns = [
        r'(?:answer|sql)[\s:]*\n*```(?:sql)?\n*(.*?)```',  # Markdown code block
        r'(?:answer|sql)[\s:]*\n*(.+?)(?:\n\n|$)',  # After "answer:" or "sql:"
        r'(SELECT.*?(?:;|$))',  # Raw SELECT statement
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                return match.group(1).strip()
            except IndexError:
                return match.group(0).strip()

    # If no pattern matches, return empty string
    return text.strip()


def create_prompt(db_schema: str, question: str) -> str:
    """Create the prompt for the model using Gemma instruct format"""
    return f"""<start_of_turn>user
Based on this database schema:

{db_schema}

Write a SQL query to answer this question. Output ONLY the SQL query, nothing else.

Question: {question}<end_of_turn>
<start_of_turn>model
"""


def evaluate_text_to_sql(
        modules: Any,
        data_loader: Any,
        num_samples: int = 10,
        max_new_tokens: int = 200,
        verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate the model on text-to-SQL task using the simplified RL dataloader format.
    """
    results = []
    correct = 0
    total = 0

    iterator = iter(data_loader)

    for i in range(num_samples):
        try:
            batch = next(iterator)
        except StopIteration:
            print(f"Data loader exhausted after {i} samples")
            break

        # Extract data (batch size 1, so take index [0] to get the string from the array)
        prompt = str(batch['prompt'][0])
        ground_truth = str(batch['ground_truth'][0])

        # Safely extract db_id if you added it back to the dataloader, otherwise default
        db_id = str(batch['db_id'][0]) if 'db_id' in batch else "Unknown"

        # Generate using the factory's sampler
        # Note: If modules is a pure tuple/dict, adjust the call. Assuming modules.sampler exists.
        generated_text = modules.sampler.sample(prompt, max_new_tokens=max_new_tokens)

        # Extract SQL from the full response using your validator logic
        predicted_sql = extract_sql_from_response(generated_text)

        # Check if correct (Wrapped in try/except in case model outputs unparseable garbage)
        try:
            is_correct = sql_compare.compare(predicted_sql, ground_truth)
        except Exception:
            is_correct = False

        if is_correct:
            correct += 1
        total += 1

        # Store result
        result = {
            'index': i,
            'db_id': db_id,
            'ground_truth': ground_truth,
            'predicted_sql': predicted_sql,
            'generated_text': generated_text,
            'is_correct': is_correct,
        }
        results.append(result)

        # Print progress
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Sample {i + 1}/{num_samples} | DB: {db_id}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Predicted: {predicted_sql}")
            print(f"✓ CORRECT" if is_correct else "✗ WRONG")

    accuracy = correct / total if total > 0 else 0.0

    print(f"\n{'=' * 80}")
    print(f"FINAL RESULTS: {correct}/{total} correct ({accuracy * 100:.1f}%)")

    return {
        'correct': correct,
        'total': total,
        'accuracy': accuracy,
        'results': results,
    }


def sql_correctness_reward(prompts: list[str], completions: list[str], ground_truth: list[str], **kwargs) -> list[
    float]:
    scores = []

    for response, target_sql in zip(completions, ground_truth):
        predicted_sql = extract_sql_from_response(response)

        if not predicted_sql:
            scores.append(0.0)
            continue
        is_valid = is_valid_syntax(predicted_sql)

        try:
            is_correct = sql_compare.compare(predicted_sql, target_sql)
        except Exception:
            is_correct = False

        if is_correct:
            scores.append(1.0)
        elif is_valid:
            scores.append(0.4)
        else:
            scores.append(0.2)

    return scores