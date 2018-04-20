import unittest
from nlp.document import ClassDocument
from nlp.naive_bayes import NaiveBayes
from math import log


class TestNaiveBayes(unittest.TestCase):

    def test_vocabulary(self):
        documents = [
            ClassDocument("I like this product very much.", "+"),
            ClassDocument("Easy to use, recommended.", "+"),
            ClassDocument("Don't like it, hard to use. Not recommended.", "-"),
        ]
        nb = NaiveBayes(documents)
        expected_vocabulary = {
            "i": 1, "like": 2, "this": 1, "product": 1, "very": 1, "much": 1,
            "easy": 1, "to": 2, "use": 2, "recommended": 2,
            "don't": 1, "it": 1, "hard": 1, "not": 1
        }
        self.assertEqual(nb.vocabulary, expected_vocabulary)

    def test_class_vocabularies(self):
        documents = [
            ClassDocument("I like this product very much.", "+"),
            ClassDocument("Easy to use, recommended.", "+"),
            ClassDocument("Don't like it, hard to use. Not recommended.", "-"),
        ]
        nb = NaiveBayes(documents)
        expected_class_vocabularies = {
            "+": {
                "i": 1, "like": 1, "this": 1, "product": 1, "very": 1, "much": 1, "easy": 1, "to": 1, "use": 1,
                "recommended": 1
            },
            "-": {
                "don't": 1, "like": 1, "it": 1, "hard": 1, "to": 1, "use": 1, "not": 1, "recommended": 1
            }
        }
        self.assertEqual(nb.class_vocabularies, expected_class_vocabularies)

    def test_class_word_counts(self):
        documents = [
            ClassDocument("I like this product very much.", "+"),
            ClassDocument("Easy to use, recommended.", "+"),
            ClassDocument("Don't like it, hard to use. Not recommended.", "-"),
        ]
        nb = NaiveBayes(documents)
        expected_class_word_counts = {"+": 10, "-": 8}
        self.assertEqual(nb.class_word_counts, expected_class_word_counts)

    def test_priors(self):
        documents = [
            ClassDocument("I like this product very much.", "+"),
            ClassDocument("Easy to use, recommended.", "+"),
            ClassDocument("Don't like it, hard to use. Not recommended.", "-"),
        ]
        nb = NaiveBayes(documents)
        self.assertEqual(nb.priors, {"+": log(2 / 3), "-": log(1 / 3)})

    def test_likelihood(self):
        documents = [
            ClassDocument("I like this product very much.", "+"),
            ClassDocument("Easy to use, recommended.", "+"),
            ClassDocument("Don't like it, hard to use. Not recommended.", "-"),
        ]
        nb = NaiveBayes(documents)
        self.assertEqual(nb.get_likelihood("use", "+"), log((1 + 1) / (10 + 14 * 1)))
        self.assertEqual(nb.get_likelihood("use", "-"), log((1 + 1) / (8 + 14 * 1)))


