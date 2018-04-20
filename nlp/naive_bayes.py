from math import log
from nlp.classifier import Classifier
import sys
from nlp.dataset import Dataset


class NaiveBayes(Classifier):

    """A Naive Bayes classifier;
    given the documents for training, it generates lists and dictionaries for probability estimation."""

    def __init__(self, documents, k=1):
        """It initializes the classifier given the documents for training and computes the priors;
        it generates the vocabulary, the vocabularies and word counts for each class.
        An optional constant is used for (Laplace) additive smoothing."""
        Classifier.__init__(self, documents)
        self.vocabulary = self.get_vocabulary(documents)
        self.class_vocabularies = self.get_class_vocabularies(documents)
        self.class_word_counts = self.get_class_word_counts(self.class_vocabularies)
        self.priors = self.get_priors()
        self.k = k

    def get_vocabulary(self, documents):
        """Returns a dictionary containing the words found (uniquely) in the documents and the number of occurrences."""
        vocabulary = {}
        for document in documents:
            for word in document.bag_of_words:
                if word not in vocabulary:
                    vocabulary[word] = 0
                vocabulary[word] += 1
        return vocabulary

    def get_class_vocabularies(self, documents):
        """Returns a dictionary for each class with the vocabulary for the documents of that class."""
        class_vocabularies = {}
        for document in documents:
            c = document.c
            if c not in class_vocabularies:
                class_vocabularies[c] = {}
            for word in document.bag_of_words:
                if word not in class_vocabularies[c]:
                    class_vocabularies[c][word] = 0
                class_vocabularies[c][word] += 1
        return class_vocabularies

    def get_class_word_counts(self, class_vocabularies):
        """Returns a dictionary for each class containing the word counts for the class given the class vocabularies."""
        class_word_counts = {}
        for c in class_vocabularies:
            class_word_counts[c] = 0
            for word in class_vocabularies[c]:
                class_word_counts[c] += class_vocabularies[c][word]
        return class_word_counts

    def get_priors(self):
        """Returns a dictionary for each class containing the (log) prior probability (e.g. P(c));
        calculated as the ratio of documents for a class and the total number of documents."""
        priors = {}
        for c in self.classes:
            count = 0
            total = 0
            for document in self.documents:
                k = 1
                if document.c == c:
                    count += k
                total += k
            priors[c] = log(count / total)
        return priors

    def get_likelihood(self, word, c):
        """Returns the (log) likelihood of the given word belonging to the given class (e.g. P(word | class));
        calculated as (count(word, c) + k) / (sum_(for each word in vocabulary) count(word, c) + k)"""
        count = self.k
        if word in self.class_vocabularies[c]:
            count += self.class_vocabularies[c][word]
        total = self.class_word_counts[c] + len(self.vocabulary) * self.k
        result = log(count / total)
        return result

    def get_prediction(self, testing_document):
        """Returns the predicted class for the given document;
        calculated as class that maximises the posterior probability (e.g. P(class | word))."""
        best_class = None
        best_class_posterior = None
        for c in self.classes:
            prior = self.priors[c]
            likelihood = 0
            for word in testing_document.bag_of_words:
                word_likelihood = self.get_likelihood(word, c)
                likelihood += word_likelihood
            if best_class is None or prior + likelihood > best_class_posterior:
                best_class_posterior = prior + likelihood
                best_class = c
        return best_class













































