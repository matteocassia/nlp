import unittest
from src.document import Document


class TestDocument(unittest.TestCase):

    def test_get_ngrams(self):
        ngrams = [2, 3]
        bag_of_words = Document.get_ngrams(ngrams, ["hello", "this", "is", "a", "document"])
        expected_bag_of_words = [
            "hello this", "this is", "is a", "a document",
            "hello this is", "this is a", "is a document"
        ]
        self.assertEqual(bag_of_words, expected_bag_of_words)

    def test_get_bag_without_stopwords(self):
        bag_of_words = Document.get_bag_without_stopwords(["the", "is", "an", "article", "and", "so", "a"])
        expected_bag_of_words = ["is", "article", "so"]
        self.assertEqual(bag_of_words, expected_bag_of_words)

    def test_get_parsed_text(self):
        parsed_text = Document.get_parsed_text("Just trying to check that: 1. numbers are removed and 2. punctuation doesn't stay.")
        expected_parsed_text = "just trying to check that numbers are removed and punctuation doesn't stay"
        self.assertEqual(parsed_text, expected_parsed_text)

    def test_get_bag_of_words(self):
        bag_of_words = Document.get_bag_of_words("sample parsed text to split on whitespace")
        expected_bag_of_words = ["sample", "parsed", "text", "to", "split", "on", "whitespace"]
        self.assertEqual(bag_of_words, expected_bag_of_words)

