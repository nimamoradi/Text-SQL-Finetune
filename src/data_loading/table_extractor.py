
import re
import os

def get_create_table_blocks(file_path: str) -> list[str]:
    """
    Reads an SQL file and extracts all complete 'CREATE TABLE' statements.

    This function is designed to handle various SQL formatting styles,
    including multi-line statements and different table name quoting conventions
    (backticks, double quotes, or unquoted standard identifiers).

    Args:
        file_path: The absolute path to the SQL file.

    Returns:
        A list of strings, where each string is a complete 'CREATE TABLE' statement
        (including the terminating semicolon).
        Returns an empty list if the file does not exist, cannot be read,
        or contains no 'CREATE TABLE' statements.
    """
    create_table_statements = []


    pattern = re.compile(
        r"CREATE\s+TABLE\s+"
        r"(?:`[^`]+`|\"[^\"]+\"|[a-zA-Z_][a-zA-Z0-9_]*)" # Table name (quoted or unquoted)
        r"\s*\(" # Opening parenthesis for table definition
        r".*?" # Non-greedy match for everything inside the parentheses
        r"\);" # Closing parenthesis and statement terminator semicolon
        , re.IGNORECASE | re.DOTALL
    )


    # Ensure the file exists before attempting to open
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        create_table_statements = pattern.findall(content)


    return create_table_statements