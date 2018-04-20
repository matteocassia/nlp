import unittest
from src.utilities import ConfusionMatrix


class TestConfusionMatrix(unittest.TestCase):

    def test_get_initial_matrix(self):
        confusion_matrix = ConfusionMatrix(["+", "-"])
        expected_matrix = [[0, 0], [0, 0]]
        self.assertEqual(confusion_matrix.matrix, expected_matrix)

    def test_add(self):
        confusion_matrix = ConfusionMatrix(["+", "-"])
        confusion_matrix.add("+", "+")
        self.assertEqual(confusion_matrix.matrix, [[1, 0], [0, 0]])
        confusion_matrix.add("+", "-")
        self.assertEqual(confusion_matrix.matrix, [[1, 0], [1, 0]])
        confusion_matrix.add("-", "-")
        self.assertEqual(confusion_matrix.matrix, [[1, 0], [1, 1]])
        confusion_matrix.add("-", "+")
        self.assertEqual(confusion_matrix.matrix, [[1, 1], [1, 1]])

    def test_get_accuracy(self):
        confusion_matrix = ConfusionMatrix(["+", "-"])
        confusion_matrix.add("+", "+")
        confusion_matrix.add("+", "-")
        confusion_matrix.add("-", "-")
        confusion_matrix.add("-", "+")
        self.assertEqual(confusion_matrix.get_accuracy(), 2 / 4)
        confusion_matrix.add("-", "+")
        self.assertEqual(confusion_matrix.get_accuracy(), 2 / 5)
        confusion_matrix.add("-", "+")
        self.assertEqual(confusion_matrix.get_accuracy(), 2 / 6)



