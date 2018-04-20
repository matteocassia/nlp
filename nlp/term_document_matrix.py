from math import log
from nlp.utilities import Process


class TermDocumentMatrix:

    def __init__(self, documents, compute_word_vectors=True, compute_document_vectors=True, minimum_word_count=0, log=False):
        """Initialises a term-document matrix object with tf-idf weights for the given documents.

        A vocabulary is always generated with the words appearing more than minimum_word_count times;
        word vectors and document vectors are always generated unless it is otherwise specified."""
        self.documents = documents
        self.log = log
        self.inverse_document_frequencies = {}
        word_counts = TermDocumentMatrix.get_word_counts(documents)
        self.vocabulary = TermDocumentMatrix.get_vocabulary(word_counts, minimum=minimum_word_count)
        if compute_document_vectors:
            self.document_vectors = self.get_document_vectors(documents)
        if compute_word_vectors:
            self.word_vectors = self.get_word_vectors(documents)

    def get_word_vectors(self, documents):
        """Returns a list of size len(vocabulary) containing tuples of size len(documents)
        containing the tf-idf weights of the words for the documents. The nth entry in the
        list of vectors is the word vector for the nth word in the vocabulary."""
        word_vectors = []
        if self.log: p = Process("Generating word vectors", len(self.vocabulary), 1)
        for word in self.vocabulary:
            if self.log: p.add()
            word_vector = []
            idf = self.get_inverse_document_frequency(word)
            for document in documents:
                tf = TermDocumentMatrix.get_term_frequency(word, document)
                word_vector.append(tf * idf)
            word_vectors.append(tuple(word_vector))
        if self.log: p.finish()
        return word_vectors

    def get_document_vectors(self, documents):
        """Returns the vectors of the given documents as a list of size len(documents)
        where the nth entry is a dictionary (word, tf-idf weight) for the nth document."""
        vectors = []
        if self.log: p = Process("Generating document vectors", len(documents), 1)
        for document in documents:
            if self.log: p.add()
            vectors.append(self.get_document_vector(document))
        if self.log: p.finish()
        return vectors

    def get_document_vector(self, document):
        """Returns a dictionary (word, tf-idf weight) for the given document."""
        vector = {}
        for word in document.bag_of_words:
            if word in self.vocabulary and word not in vector:
                tf = TermDocumentMatrix.get_term_frequency(word, document)
                idf = self.get_inverse_document_frequency(word)
                vector[word] = tf * idf
        return vector

    def get_inverse_document_frequency(self, word):
        """Returns the inverse document frequency of the given word."""
        if word in self.inverse_document_frequencies:
            return self.inverse_document_frequencies[word]
        else:
            count = 0
            for document in self.documents:
                if word in document.bag_of_words:
                    count += 1
            self.inverse_document_frequencies[word] = log(len(self.documents) / (count + 1))
            return self.get_inverse_document_frequency(word)

    @staticmethod
    def get_term_frequency(word, document):
        """Returns the term frequency of the given word in the given document."""
        if word not in document.bag_of_words:
            return 0
        elif type(document.bag_of_words) is set:
            return 1
        else:
            return document.bag_of_words.count(word)

    @staticmethod
    def get_vocabulary(word_counts, minimum=0):
        """Returns the vocabulary as a list of word for the given word counts;
        only words whose count is higher than the optional parameter minimum are retained."""
        vocabulary = []
        for word in word_counts:
            if word_counts[word] > minimum and word not in vocabulary:
                vocabulary.append(word)
        return vocabulary

    @staticmethod
    def get_word_counts(documents, log=False):
        """Returns a dictionary (word, count) of the word counts for each word
        in the given documents."""
        word_counts = {}
        p = None
        if log:
            p = Process("Generating word counts", len(documents), 1)
        for document in documents:
            if log:
                p.add()
            for word in document.bag_of_words:
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1
        if log:
            p.finish()
        return word_counts