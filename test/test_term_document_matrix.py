import unittest
from src.term_document_matrix import TermDocumentMatrix
from src.document import Document
from math import log


class TestTermDocumentMatrix(unittest.TestCase):

    def test_get_term_frequency(self):
        document1 = Document("This document is a document with no duplicates.", preserve_duplicates=False)
        document2 = Document("This document is a document with duplicates.", preserve_duplicates=True)
        document3 = Document("This is just to check that a word is not present.", preserve_duplicates=True)
        self.assertEqual(TermDocumentMatrix.get_term_frequency("document", document1), 1)
        self.assertEqual(TermDocumentMatrix.get_term_frequency("document", document2), 2)
        self.assertEqual(TermDocumentMatrix.get_term_frequency("document", document3), 0)

    def test_get_inverse_document_frequency(self):
        documents = [
            Document("This document is a document with no duplicates."),
            Document("This document is a document with duplicates."),
            Document("This is just to check that a word is not present.")
        ]
        term_document_matrix = TermDocumentMatrix(documents)
        self.assertEqual(term_document_matrix.get_inverse_document_frequency("document"), log(3 / 3))
        self.assertEqual(term_document_matrix.get_inverse_document_frequency("not_present"), log(3 / 1))
        self.assertEqual(term_document_matrix.get_inverse_document_frequency("just"), log(3 / 2))
        self.assertEqual(term_document_matrix.get_inverse_document_frequency("is"), log(3 / 4))

    def test_get_word_counts(self):
        documents = [
            Document("This document is a document with no duplicates."),
            Document("This document is a document with duplicates."),
            Document("This is just to check that a word is not present.")
        ]
        expected_word_counts = {
            "this": 3,
            "document": 2,
            "is": 3,
            "a": 3,
            "with": 2,
            "no": 1,
            "duplicates": 2,
            "just": 1,
            "to": 1,
            "check": 1,
            "that": 1,
            "word": 1,
            "not": 1,
            "present": 1
        }
        word_counts = TermDocumentMatrix.get_word_counts(documents)
        self.assertEqual(word_counts, expected_word_counts)

    def test_get_vocabulary(self):
        word_counts = {
            "this": 3,
            "document": 2,
            "is": 3,
            "a": 3,
            "with": 2,
            "no": 1,
            "duplicates": 2,
            "just": 1,
            "to": 1,
            "check": 1,
            "that": 1,
            "word": 1,
            "not": 1,
            "present": 1
        }
        expected_vocabulary = ["this", "document", "is", "a", "with", "no", "duplicates", "just", "to", "check", "that",
                               "word", "not", "present"]
        vocabulary = TermDocumentMatrix.get_vocabulary(word_counts)
        self.assertEqual(set(expected_vocabulary), set(vocabulary))

    def test_get_document_vector(self):
        documents = [
            Document("Nice document"),
            Document("Bad document"),
            Document("Alright document"),
            Document("Nice day today"),
            Document("No day is a bad day")
        ]
        term_document_matrix = TermDocumentMatrix(documents)
        document = Document("Today is a bad, bad day", preserve_duplicates=True)
        document_vector = term_document_matrix.get_document_vector(document)
        expected_document_vector = {
            "today": 1 * log(5 / 2),
            "is": 1 * log(5 / 2),
            "a": 1 * log(5 / 2),
            "bad": 2 * log(5 / 3),
            "day": 1 * log(5 / 3)
        }
        self.assertEqual(document_vector, expected_document_vector)

    def test_get_word_vector(self):
        documents = [
            Document("Nice document, document", preserve_duplicates=True),
            Document("Bad document"),
            Document("Alright document"),
            Document("Nice day today"),
            Document("No day is a bad day")
        ]
        term_document_matrix = TermDocumentMatrix(documents)
        word_vector = term_document_matrix.word_vectors[term_document_matrix.vocabulary.index("document")]
        expected_word_vector = [
            2 * log(5 / 4), 1 * log(5 / 4), 1 * log(5 / 4), 0.0, 0.0
        ]
        self.assertEqual(word_vector, tuple(expected_word_vector))



