from nlp.term_document_matrix import TermDocumentMatrix
from math import pow, sqrt, log
from nlp.classifier import Classifier


class KNN(Classifier):

    """A KNN classifier; given the training documents and number of neighbours, it generates the document vectors."""

    def __init__(self, documents, k, weight_neighbors=True):
        """Initializes the classifier given the training documents and number of neighbours;
        it computes the term-document matrix necessary to obtain the vocabulary and tf-idf document vectors."""
        Classifier.__init__(self, documents)
        self.term_document_matrix = TermDocumentMatrix(documents, compute_word_vectors=False)
        self.document_vectors = self.term_document_matrix.document_vectors
        self.k = k
        self.weight_neighbors = weight_neighbors

    def get_prediction(self, document):
        """Returns the predicted class for the given document;
        it is done my finding the k most similar documents and computing the majority class."""
        most_similar_documents, similarities = self.get_most_similar_documents(document)
        if self.weight_neighbors:
            return KNN.get_majority_class(most_similar_documents, similarities)
        else:
            return KNN.get_simple_majority_class(most_similar_documents)

    def get_most_similar_documents(self, target):
        """Returns the k most similar documents to the given document;
        it is done by computing the cosine similarity between each training document and the target document."""
        target_vector = self.term_document_matrix.get_document_vector(target)
        documents_and_similarities = []
        for i in range(0, len(self.documents)):
            similarity = KNN.get_similarity(target_vector, self.document_vectors[i])
            documents_and_similarities.append((self.documents[i], similarity))
        documents_and_similarities.sort(key=lambda t: t[1], reverse=True)
        documents_and_similarities = documents_and_similarities[0:self.k]
        most_similar_documents = []
        similarities = []
        for x in documents_and_similarities:
            most_similar_documents.append(x[0])
            similarities.append(x[1])
        return most_similar_documents, similarities

    @staticmethod
    def get_majority_class(documents, similarities):
        """Returns the weighted majority class given the neighbour documents and their similarities."""
        scores = {}
        minimum, maximum = KNN.get_minimum_and_maximum(similarities)
        for i in range(0, len(documents)):
            document = documents[i]
            similarity = KNN.get_normalised_similarity(minimum, maximum, 1, 2, similarities[i])
            c = document.c
            if c not in scores:
                scores[c] = 0
            scores[c] += similarity
        majority_class = None
        majority_class_score = -1
        for c in scores:
            if majority_class is None or scores[c] > majority_class_score:
                majority_class = c
                majority_class_score = scores[c]
        return majority_class

    @staticmethod
    def get_minimum_and_maximum(similarities):
        """Returns the minimum and maximum values found in the given list of similarities."""
        minimum = float("+inf")
        maximum = float("-inf")
        for similarity in similarities:
            minimum = min(minimum, similarity)
            maximum = max(maximum, similarity)
        return minimum, maximum

    @staticmethod
    def get_normalised_similarity(minimum, maximum, beta, alpha, similarity):
        """Returns the normalised similarity given the original similarity, the minimum and maximum similarity
        of the other neighbours and the minimum and maximum of the normalisation scale."""
        if maximum != minimum:
            return beta + (alpha - beta) * (similarity - minimum) / (maximum - minimum)
        else:
            return beta

    @staticmethod
    def get_simple_majority_class(documents):
        """Returns the majority class of the given documents."""
        counts = {}
        for document in documents:
            if document.c not in counts:
                counts[document.c] = 0
            counts[document.c] += 1
        majority_class = None
        majority_class_count = -1
        for c in counts:
            if counts[c] > majority_class_count:
                majority_class = c
                majority_class_count = counts[c]
        return majority_class

    @staticmethod
    def get_magnitude(document_vector):
        """Returns the magnitude of the given vector."""
        sum = 0
        for word in document_vector:
            sum += pow(document_vector[word], 2)
        return sqrt(sum)

    @staticmethod
    def get_dot_product(document_vector_a, document_vector_b):
        """Returns the dot product of the two given given vectors."""
        dot_product = 0
        for word in document_vector_a:
            if word in document_vector_b:
                dot_product += document_vector_a[word] * document_vector_b[word]
        return dot_product

    @staticmethod
    def get_similarity(document_vector_a, document_vector_b):
        """Returns the cosine similarity of the two given vectors."""
        dot_product = KNN.get_dot_product(document_vector_a, document_vector_b)
        squared_sum_a = KNN.get_magnitude(document_vector_a)
        squared_sum_b = KNN.get_magnitude(document_vector_b)
        if squared_sum_a != 0 and squared_sum_b != 0:
            return dot_product / (squared_sum_a * squared_sum_b)
        else:
            return 0