import unittest
import os
import sys

# Ensure project root is in the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_loading.sql_comparator import (
    SQLCompareAdapter,
    TransformingSQLComparator,
)


class TestSQLCompareAdapter(unittest.TestCase):
    def setUp(self):
        self.adapter = SQLCompareAdapter()

    def test_basic_equality(self):
        self.assertTrue(
            self.adapter.compare(
                "SELECT * FROM users", "SELECT * FROM users"
            )
        )

    def test_case_mismatch(self):
        # sql_compare.compare is case-sensitive for non-keyword tokens
        self.assertFalse(
            self.adapter.compare(
                "SELECT count(*) FROM club", "SELECT COUNT(*) FROM club"
            )
        )


class TestTransformingSQLComparator(unittest.TestCase):
    def setUp(self):
        self.comparator = TransformingSQLComparator()

    def test_transform_lowercase(self):
        sql = "SELECT * FROM Users"
        self.assertEqual(self.comparator.transform(sql), "select * from users")

    def test_transform_single_line(self):
        sql = "SELECT *\nFROM\nusers"
        self.assertEqual(self.comparator.transform(sql), "select * from users")

    def test_transform_semicolon_removal(self):
        sql = "SELECT * FROM users;"
        self.assertEqual(self.comparator.transform(sql), "select * from users")

    def test_transform_quotes_removal(self):
        sql = "SELECT * FROM \"users\" WHERE name = `John` AND name2 = 'Doe'"
        self.assertEqual(
            self.comparator.transform(sql),
            "select * from users where name=john and name2=doe",
        )

    def test_user_examples(self):
        # The specific examples mentioned by the user should pass
        gt = "SELECT count(*) FROM club"

        # Sample 1
        pred1 = 'SELECT COUNT(*) FROM "club";'
        self.assertTrue(self.comparator.compare(pred1, gt))

        # Sample 2
        pred2 = 'SELECT COUNT(*)\nFROM "club";'
        self.assertTrue(self.comparator.compare(pred2, gt))

    def test_new_user_example(self):
        expected = 'SELECT population FROM state WHERE state_name  =  "california";'
        predicted = (
            "SELECT population\nFROM `state`\nWHERE `state_name` = 'California';"
        )
        self.assertTrue(self.comparator.compare(predicted, expected))

    def test_transform_alias_removal(self):
        sql = "select team as t from club"
        self.assertEqual(self.comparator.transform(sql), "select team from club")
        sql2 = "select team as t1 from club as c"
        self.assertEqual(self.comparator.transform(sql2), "select team from club c")

    def test_transform_punctuation_spacing(self):
        sql = "select count(*) , sum(points) from club where a = b"
        self.assertEqual(
            self.comparator.transform(sql),
            "select count(*),sum(points)from club where a=b",
        )

    def test_user_rename_queries(self):
        expected_with_as = "SELECT T1.DName FROM DEPARTMENT AS T1 JOIN MEMBER_OF AS T2 ON T1.DNO  =  T2.DNO GROUP BY T2.DNO ORDER BY count(*) ASC LIMIT 1"
        expected_without_as = "SELECT T1.DName FROM DEPARTMENT T1 JOIN MEMBER_OF T2 ON T1.DNO  =  T2.DNO GROUP BY T2.DNO ORDER BY count(*) ASC LIMIT 1"
        
        # Test they resolve to the exact same transformed string
        t1 = self.comparator.transform(expected_with_as)
        t2 = self.comparator.transform(expected_without_as)
        self.assertEqual(t1, t2)
        self.assertTrue(self.comparator.compare(expected_with_as, expected_without_as))

        # Test expected and predicted are not equal (completely different queries)
        predicted = (
            "SELECT DISTINCT s.Name\n"
            "FROM Student s\n"
            "INNER JOIN Faculty f ON s.StuID = f.FacID\n"
            "INNER JOIN Department d ON f.Lname = d.Lname\n"
            "WHERE d.DName = 'Department';"
        )
        self.assertFalse(self.comparator.compare(predicted, expected_with_as))


if __name__ == "__main__":
    unittest.main()
