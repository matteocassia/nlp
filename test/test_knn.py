import unittest
from math import sqrt
from src.knn import KNN
from src.document import ClassDocument


class TestKNN(unittest.TestCase):

    def test_get_magnitude(self):
        self.assertEqual(KNN.get_magnitude({"hello": 1, "my": 1, "friend": 3}), sqrt(11))
        self.assertEqual(KNN.get_magnitude({"hello": 1, "my": 1, "enemy": 2}), sqrt(6))

    def test_get_dot_product(self):
        document_vector_a = {"hello": 1, "my": 1, "friend": 3}
        document_vector_b = {"hello": 1, "my": 1, "enemy": 2}
        expected_dot_product = 2
        self.assertEqual(KNN.get_dot_product(document_vector_a, document_vector_b), expected_dot_product)

    def test_get_similarity(self):
        document_vector_a = {"hello": 1, "my": 1, "friend": 3}
        document_vector_b = {"hello": 1, "my": 1, "enemy": 2}
        self.assertEqual(KNN.get_similarity(document_vector_a, document_vector_b), 2 / (sqrt(11) * sqrt(6)))

    def test_get_simple_majority_class(self):
        documents = [
            ClassDocument("A positive document.", "+"),
            ClassDocument("Another positive document.", "+"),
            ClassDocument("A negative document instead.", "-")
        ]
        self.assertEqual(KNN.get_simple_majority_class(documents), "+")

    def test_get_minimum_and_maximum(self):
        similarities = [0.14, 0.01, 0.98, 0.43]
        self.assertEqual(KNN.get_minimum_and_maximum(similarities), (0.01, 0.98))

    def test_get_normalised_similarity(self):
        minimum = 0.1
        maximum = 0.7
        beta = 1.0
        alpha = 2.0
        self.assertEqual(KNN.get_normalised_similarity(minimum, maximum, beta, alpha, 0.2), 7 / 6)
        self.assertEqual(KNN.get_normalised_similarity(minimum, maximum, beta, alpha, 0.3), 4 / 3)
        self.assertEqual(KNN.get_normalised_similarity(minimum, maximum, beta, alpha, 0.4), 3 / 2)

    def test_get_majority_class(self):
        documents = [
            ClassDocument("pos", "+"),
            ClassDocument("pos", "+"),
            ClassDocument("neg", "-"),
            ClassDocument("neg", "-"),
            ClassDocument("neg", "-")
        ]
        similarities = [0.9, 0.8, 0.3, 0.2, 0.1]
        self.assertEqual(KNN.get_majority_class(documents, similarities), "+")

