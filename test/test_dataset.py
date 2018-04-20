import unittest
from nlp.dataset import Dataset, ClassDataset
from nlp.document import ClassDocument


class TestDataset(unittest.TestCase):

    def test_get_documents_by_class(self):
        documents = [
            ClassDocument("A positive document", "+"),
            ClassDocument("Another positive document", "+"),
            ClassDocument("Yet one more positive document", "+"),
            ClassDocument("This time a negative one", "-"),
            ClassDocument("Still negative", "-")
        ]
        documents_by_class = Dataset.get_documents_by_class(documents)
        expected_documents_by_class = {
            "+": [documents[0], documents[1], documents[2]],
            "-": [documents[3], documents[4]]
        }
        self.assertEqual(documents_by_class, expected_documents_by_class)

    def test_get_minimum_class_count(self):
        documents = [
            ClassDocument("A positive document", "+"),
            ClassDocument("Another positive document", "+"),
            ClassDocument("Yet one more positive document", "+"),
            ClassDocument("This time a negative one", "-"),
            ClassDocument("Still negative", "-")
        ]
        documents_by_class = {
            "+": [documents[0], documents[1], documents[2]],
            "-": [documents[3], documents[4]]
        }
        minimum_class_count = Dataset.get_minimum_class_count(documents_by_class)
        expected_minimum_class_count = 2
        self.assertEqual(minimum_class_count, expected_minimum_class_count)

    def test_get_normalised_documents(self):
        documents = [
            ClassDocument("A positive document", "+"),
            ClassDocument("Another positive document", "+"),
            ClassDocument("Yet one more positive document", "+"),
            ClassDocument("This time a negative one", "-"),
            ClassDocument("Still negative", "-")
        ]
        normalised_documents = ClassDataset.get_normalised_documents(documents)
        expected_normalised_documents = [documents[0], documents[1], documents[3], documents[4]]
        self.assertEqual(normalised_documents, expected_normalised_documents)

    def test_get_split_documents(self):
        documents = [
            ClassDocument("pos1", "+"),
            ClassDocument("pos2", "+"),
            ClassDocument("pos3", "+"),
            ClassDocument("pos4", "+"),
            ClassDocument("pos5", "+"),
            ClassDocument("neg1", "-"),
            ClassDocument("neg2", "-"),
            ClassDocument("neg3", "-"),
            ClassDocument("neg4", "-"),
            ClassDocument("neg5", "-"),
        ]
        split_documents = Dataset.get_split_documents(documents, 0.8)
        expected_split_documents = [
            documents[0], documents[1], documents[2], documents[3],
            documents[5], documents[6], documents[7], documents[8]
        ], [documents[4], documents[9]]
        self.assertEqual(split_documents, expected_split_documents)

