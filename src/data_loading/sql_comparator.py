from abc import ABC, abstractmethod
import re
import sql_compare


class SQLComparator(ABC):
    """Abstract Base Class (Interface) for comparing SQL queries."""

    @abstractmethod
    def compare(self, predicted: str, reference: str) -> bool:
        """Compare predicted SQL query with reference SQL query."""
        pass


class SQLCompareAdapter(SQLComparator):
    """Adapter that wraps the original sql_compare library directly."""

    def compare(self, predicted: str, reference: str) -> bool:
        try:
            ans = sql_compare.compare(predicted, reference)
            return ans
        except Exception:
            return False


class TransformingSQLComparator(SQLComparator):
    """
    SQL Comparator that transforms both SQL strings using a simple transformation
    method before comparing them.
    """

    def __init__(self, delegate_comparator: SQLComparator = None):
        self.delegate_comparator = delegate_comparator or SQLCompareAdapter()

    def transform(self, sql: str) -> str:
        """Transform a SQL string to make it standard and alike.

        Performs:
        1. Convert all words to lower case.
        2. Put all text to a single line.
        3. Isolate SELECT clause and remove "AS alias" patterns from it.
        4. Remove "AS" keyword from the entire SQL query (keeping table alias names).
        5. Remove single quotes, double quotes, and backticks.
        6. Normalize whitespace around punctuation and operators.
        7. Remove semicolon at the end of the string.
        """
        if not sql:
            return ""

        # 1. Convert all words to lower case
        sql_transformed = sql.lower()

        # 2. Put all text to a single line
        sql_transformed = re.sub(r"\s+", " ", sql_transformed).strip()

        # 3. Handle SELECT clause aliases specifically to strip "AS alias" from select fields
        parts = sql_transformed.split(" from ", 1)
        if len(parts) == 2:
            select_clause, rest = parts
            select_clause = re.sub(r"\b(as)\s+\w+\b", "", select_clause)
            sql_transformed = select_clause + " from " + rest
        else:
            sql_transformed = re.sub(r"\b(as)\s+\w+\b", "", sql_transformed)

        # 4. Remove "AS" keyword from the entire SQL query (keeps table alias names intact)
        sql_transformed = re.sub(r"\b(as)\b", "", sql_transformed)

        # 5. Remove single quotes, double quotes, and backticks
        sql_transformed = (
            sql_transformed.replace("'", "").replace('"', "").replace("`", "")
        )

        # 6. Normalize whitespace around punctuation and operators
        sql_transformed = re.sub(r"\s*([,=\(\)<>!])\s*", r"\1", sql_transformed)

        # 7. Collapse remaining double spaces and strip
        sql_transformed = re.sub(r"\s+", " ", sql_transformed).strip()

        # 8. Remove semicolon at the end of the string
        sql_transformed = sql_transformed.rstrip(";").strip()

        return sql_transformed

    def compare(self, predicted: str, reference: str) -> bool:
        norm_pred = self.transform(predicted)
        norm_ref = self.transform(reference)
        return self.delegate_comparator.compare(norm_pred, norm_ref)
