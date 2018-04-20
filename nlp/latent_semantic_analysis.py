from nlp.term_document_matrix import TermDocumentMatrix
from numpy.linalg import svd
import sys
from nlp.dataset import Dataset


class LatentSemanticAnalysis:

    """A class to perform latent semantic analysis on a corpus of documents."""

    def __init__(self, documents):
        """Initializes the utility given the documents;
        the word vectors are generated along with the dictionary, then stores the singular-value decomposition."""
        self.documents = documents
        term_document_matrix = TermDocumentMatrix(documents, compute_document_vectors=False, log=True)
        self.vocabulary = term_document_matrix.vocabulary
        self.word_vectors, self.singular_values, self.document_vectors = svd(term_document_matrix.word_vectors,
                                                                             full_matrices=False)

    def get_keywords_for_topic(self, topic_number, n=10):
        """Returns a list containing the n words mostly associated to the topic at the given index."""
        pairs = []
        for i in range(0, len(self.vocabulary)):
            pairs.append((self.vocabulary[i], self.word_vectors[i][topic_number]))
        pairs.sort(key=lambda t: t[1], reverse=True)
        keywords = []
        pairs = pairs[0:n]
        for pair in pairs:
            keywords.append(pair[0])
        return keywords

    def get_keywords_for_topics(self, number_of_topics, n):
        """Returns a list containing the lists of keywords for each topic at the specified indexes."""
        keywords = []
        for topic_number in range(0, number_of_topics):
            keywords.append(self.get_keywords_for_topic(topic_number, n))
        return keywords


# lsa.py <dataset> <number_of_topics> <number_of_keywords>

if __name__ == '__main__':
    arguments = sys.argv
    format = "latent_semantic_analysis.py <dataset> <number of topics> <number of keywords>"
    if len(arguments) == 4:
        dataset_name = arguments[1]
        topics_number = arguments[2]
        keywords_number = arguments[3]
        dataset = Dataset.get_dataset(dataset_name)
        if dataset is not None:
            try:
                topics_number = int(topics_number)
                keywords_number = int(keywords_number)
            except:
                print("Invalid number format for number of topics and / or keywords; provide an integer.")
                print("  > " + format)
            lsa = LatentSemanticAnalysis(dataset.documents)
            keywords = lsa.get_keywords_for_topics(topics_number, keywords_number)
            for i in range(0, len(keywords)):
                print("Topic " + str(i + 1))
                string = ""
                for word in keywords[i]:
                    string += str(word) + " "
                print(string + "\n")
        else:
            print("Invalid dataset name.")
            print("  > " + format)
    else:
        print("Invalid number of arguments.")
        print("  > " + format)
