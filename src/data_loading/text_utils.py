import re
from typing import Dict, Any
import sql_compare
import sqlparse
from src.data_loading.sql_comparator import TransformingSQLComparator

_default_comparator = TransformingSQLComparator()


def get_default_comparator():
    return _default_comparator


def set_default_comparator(comparator):
    global _default_comparator
    _default_comparator = comparator


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
        r'```(?:sql)?\n*(.*?)```',  # Markdown code block
        r'(?:answer|sql)[\s:]*\n*```(?:sql)?\n*(.*?)```',  # Markdown code block with prefix
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
Please use English.

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
        prompt_key = 'prompt' if 'prompt' in batch else 'prompts'
        prompt = str(batch[prompt_key][0])
        ground_truth = str(batch['ground_truth'][0])

        # Safely extract db_id if you added it back to the dataloader, otherwise default
        db_id = str(batch['db_id'][0]) if 'db_id' in batch else "Unknown"

        # Generate using the factory's sampler
        # Note: If modules is a pure tuple/dict, adjust the call. Assuming modules.sampler exists.
        sampler_output = modules.sampler(prompt, max_generation_steps=max_new_tokens)
        generated_text = sampler_output.text[0]

        # Extract SQL from the full response using your validator logic
        predicted_sql = extract_sql_from_response(generated_text)

        # Check if correct (Wrapped in try/except in case model outputs unparseable garbage)
        try:
            is_correct = get_default_comparator().compare(predicted_sql, ground_truth)
        except Exception:
            is_correct = False

        if is_correct:
            correct += 1
        total += 1

        # Store result
        result = {
            'index': i,
            'db_id': db_id,
            'prompt': prompt,
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


def sql_correctness_reward(prompts: list[str], completions: list[str], ground_truth: list[str], verbose: bool = False, **kwargs) -> list[float]:
    import difflib
    scores = []
    
    # Extract decay from kwargs, default to 1.0 (no decay)
    decay = kwargs.get('decay', 1.0)

    for response, target_sql in zip(completions, ground_truth):
        predicted_sql = extract_sql_from_response(response)

        score = 0.0
        
        if not predicted_sql:
            score = -1.0 # Strong penalty for failing to generate any SQL
        else:
            is_valid = is_valid_syntax(predicted_sql)
            try:
                is_correct = get_default_comparator().compare(predicted_sql, target_sql)
            except Exception:
                is_correct = False

            # Correctness & Validity Rewards
            if is_correct:
                score = 1.0   # Max reward for exact correctness (no decay)
            else:
                # Partial credit for syntax validity
                if is_valid:
                    score += 0.2 * decay
                else:
                    score -= 0.5 * decay # Penalty for invalid syntax
                
                # Similarity reward
                sim_ratio = difflib.SequenceMatcher(None, predicted_sql, target_sql).ratio()
                score += sim_ratio * 0.5 * decay
                
                # Length similarity reward
                len_p = len(predicted_sql)
                len_t = len(target_sql)
                if len_p > 0 and len_t > 0:
                    len_ratio = min(len_p, len_t) / max(len_p, len_t)
                    score += len_ratio * 0.5 * decay
                    
        # PENALTIES
        # 1. Non-ASCII penalty (penalize generating other languages)
        if not all(ord(c) < 128 for c in response):
            score -= 1.0
            
        # 2. Verbosity penalty (model should return less conversational text)
        extra_chars = len(response) - len(predicted_sql)
        if extra_chars > 40:
            score -= 0.5
            
        # Ensure score is strictly in the range [-1.0, 1.0]
        score = max(-1.0, min(1.0, score))
                
        scores.append(score)

        if verbose:
            # Write to a file so we can see it in real-time despite the progress bar
            try:
                with open("rewards_debug.log", "a", encoding="utf-8") as f:
                    f.write(f"--- REWARD: {score:.3f} ---\n")
                    f.write(f"EXPECTED:  {target_sql}\n")
                    f.write(f"PREDICTED: {predicted_sql}\n")
                    f.write(f"RAW TEXT:  {response!r}\n\n")
            except Exception:
                pass

    return scores