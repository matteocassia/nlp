import unittest
from nlp.document import ClassDocument
from nlp.id3 import ID3
from math import log


class TestID3(unittest.TestCase):

    def test_get_entropy(self):
        documents = [
            ClassDocument("A positive document.", "+"),
            ClassDocument("Another positive document.", "+"),
            ClassDocument("Still positive.", "+"),
            ClassDocument("This time it's negative.", "-"),
            ClassDocument("Negative again.", "-"),
        ]
        id3 = ID3(documents)
        self.assertEqual(id3.get_entropy(documents), 0 - (3 / 5) * log(3 / 5, 2) - (2 / 5) * log(2 / 5, 2))

    def test_get_information_gain(self):
        documents = [
            ClassDocument("A positive document.", "+"),
            ClassDocument("Another positive document.", "+"),
            ClassDocument("Still a document.", "+"),
            ClassDocument("This time it's negative.", "-"),
            ClassDocument("Negative again.", "-"),
        ]
        id3 = ID3(documents)
        entropy1 = 0 - (3 / 5) * log(3 / 5, 2) - (2 / 5) * log(2 / 5, 2)
        entropy2 = (3 / 5) * (0 - (3 / 3) * log(3 / 3, 2) - 0)
        entropy3 = (2 / 5) * (0 - 0 - (2 / 2) * log(2 / 2, 2))
        expected_gain = entropy1 - entropy2 - entropy3
        self.assertEqual(id3.get_information_gain("document", documents), expected_gain)
        entropy1 = 0 - (3 / 5) * log(3 / 5, 2) - (2 / 5) * log(2 / 5, 2)
        entropy2 = (2 / 5) * (0 - (2 / 2) * log(2 / 2, 2) - 0)
        entropy3 = (3 / 5) * (0 - (1 / 3) * log(1 / 3, 2) - (2 / 3) * log(2 / 3, 2))
        expected_gain = entropy1 - entropy2 - entropy3
        self.assertAlmostEqual(id3.get_information_gain("a", documents), expected_gain)

    def test_get_most_informative_word(self):
        documents = [
            ClassDocument("Hello, friend!", "+"),
            ClassDocument("Hello, pal!", "+"),
            ClassDocument("Hello, you.", "+"),
            ClassDocument("Goodbye, friend.", "-"),
            ClassDocument("Bye, human.", "-"),
        ]
        id3 = ID3(documents)
        self.assertEqual(id3.get_most_informative_word(documents, id3.vocabulary), "hello")

    def test_get_split_data(self):
        documents = [
            ClassDocument("Hello, friend!", "+"),
            ClassDocument("Hello, pal!", "+"),
            ClassDocument("Hello, you.", "+"),
            ClassDocument("Goodbye, friend.", "-"),
            ClassDocument("Bye, human.", "-"),
        ]
        id3 = ID3(documents)
        expected_split_data = set([documents[0], documents[3]]), set([documents[1], documents[2], documents[4]])
        self.assertEqual(id3.get_split_data("friend", documents), expected_split_data)



