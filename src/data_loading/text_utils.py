import re
from typing import Dict, Any

def normalize_sql(sql: str) -> str:
    """Remove whitespace and normalize SQL for comparison"""
    if not sql:
        return ""
    # Remove comments
    sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    # Remove extra whitespace and convert to lowercase
    sql = ' '.join(sql.split()).lower()
    # Remove trailing semicolon
    sql = sql.rstrip(';')
    return sql


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
        temperature: float = 0.8,
        verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate the model on text-to-SQL task.

    Returns:
        Dict with 'correct', 'total', 'accuracy', and 'results' (list of dicts)
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

        # Extract data (batch size 1, so take [0])
        db_schema = str(batch['db_definitions'][0])
        question = str(batch['question'][0])
        ground_truth = str(batch['query'][0])
        db_id = str(batch['db_id'][0])

        # Create prompt
        prompt = create_prompt(db_schema, question)
        print('prompt:\n', prompt + '\n done')

        # Generate using sampler
        output = modules.sampler.sample(prompt, max_new_tokens=max_new_tokens)
        generated_text = output

        print('generated_text:\n', generated_text + '\n done')
        # Extract SQL from the full response
        predicted_sql = extract_sql_from_response(generated_text)

        # Normalize both for comparison
        norm_predicted = normalize_sql(predicted_sql)
        norm_ground_truth = normalize_sql(ground_truth)

        # Check if correct
        is_correct = norm_predicted == norm_ground_truth
        if is_correct:
            correct += 1
        total += 1

        # Store result
        result = {
            'index': i,
            'db_id': db_id,
            'question': question,
            'ground_truth': ground_truth,
            'predicted_sql': predicted_sql,
            'generated_text': generated_text,
            'is_correct': is_correct,
            'norm_predicted': norm_predicted,
            'norm_ground_truth': norm_ground_truth,
        }
        results.append(result)

        # Print progress
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Sample {i + 1}/{num_samples} | DB: {db_id}")
            print(f"Question: {question}")
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
