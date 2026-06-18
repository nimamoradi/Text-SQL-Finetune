
import re
import os
import sqlite3

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


def get_sqlite_schemas(file_path: str) -> list[str]:
    """
    Connects to an SQLite database file and extracts all complete
    'CREATE TABLE' statements from its internal schema.

    Args:
        file_path: The absolute or relative path to the .sqlite file.

    Returns:
        A list of strings, where each string is a complete 'CREATE TABLE' statement
        (including the terminating semicolon).
        Returns an empty list if the database contains no user-defined tables.

    Raises:
        FileNotFoundError: If the target file does not exist.
        sqlite3.DatabaseError: If the file exists but is not a valid SQLite database.
        sqlite3.Error: For any other database execution issues.
    """
    create_table_statements = []

    # Ensure the file exists before attempting to connect.
    # If we don't do this, sqlite3.connect() will silently create a new empty file.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Database file not found: {file_path}")

    conn = None
    try:
        # Connect to the database
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()

        # Query sqlite_master for the table creation SQL.
        # We explicitly check for 'sql IS NOT NULL' to handle internal tables safely.
        cursor.execute("""
            SELECT sql 
            FROM sqlite_master 
            WHERE type='table' 
              AND name != 'sqlite_sequence' 
              AND sql IS NOT NULL;
        """)

        # Fetch and format the statements
        rows = cursor.fetchall()
        for row in rows:
            schema_sql = row[0].strip()
            # SQLite's internal master table usually strips the trailing semicolon.
            # We append it here to ensure the output matches your expected format.
            if not schema_sql.endswith(';'):
                schema_sql += ';'

            create_table_statements.append(schema_sql)

    except sqlite3.DatabaseError as e:
        # Raises if the file is corrupted or just a text file named .sqlite
        raise sqlite3.DatabaseError(f"Invalid or corrupted SQLite database '{file_path}': {e}")

    except sqlite3.Error as e:
        # Catches other unexpected SQLite-level errors
        raise sqlite3.Error(f"A database error occurred: {e}")

    finally:
        # Ensure the connection is strictly closed even if an error is thrown
        if conn:
            conn.close()

    return create_table_statements