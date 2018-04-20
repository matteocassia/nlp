from nlp.classifier import Classifier
from nlp.term_document_matrix import TermDocumentMatrix
from math import log
from abc import ABC, abstractmethod


class TreeNode(ABC):

    """An abstract class representing a tree node."""

    def __init__(self):
        """Initializes the object."""
        pass

    def get_pad(self, n):
        """Returns a string containing n whitespaces."""
        pad = ""
        for i in range(0, n):
            pad += "  "
        return pad

    @abstractmethod
    def get_string(self, n):
        pass


class WordTreeNode(TreeNode):

    """A class representing a tree node for a word; contains reference to the children nodes."""

    def __init__(self, word):
        """Initializes the node for the word given and creates and empty array for the children nodes."""
        TreeNode.__init__(self)
        self.word = word
        self.children = []

    def get_string(self, n):
        """Returns a string representation of the node (with n whitespaces on the left)."""
        pad = self.get_pad(n)
        string = pad + self.word
        string += "\n" + self.children[0].get_string(n + 1)
        string += "\n" + self.children[1].get_string(n + 1)
        return string


class ClassTreeNode(TreeNode):

    """A class representing a tree node for the class; it is a leaf of the tree (has no children)."""

    def __init__(self, c):
        """Initializes the node with the given class."""
        TreeNode.__init__(self)
        self.c = c

    def get_string(self, n):
        """Returns a string representation of the leaf (with n whitespaces on the left)."""
        pad = self.get_pad(n)
        string = pad + self.c
        return string


class ID3(Classifier):

    """An ID3 classifier; given documents, it generates a decision tree for classification."""

    def __init__(self, documents):
        """Initializes the classifier given the documents; it obtains the vocabulary and generates the tree."""
        Classifier.__init__(self, documents)
        documents = set(documents)
        term_document_matrix = TermDocumentMatrix(documents, compute_word_vectors=False, compute_document_vectors=False)
        self.vocabulary = set(term_document_matrix.vocabulary)
        self.tree = self.get_tree(documents, self.vocabulary)

    def get_prediction(self, document):
        """Returns the predicted class for the given document based on the previously generated decision tree."""
        return self.classify(document, self.tree)

    def classify(self, document, tree):
        """Returns the predicted class for the document;
        it is done recursively by passing child nodes until leaf."""
        if type(tree) is ClassTreeNode:
            return tree.c
        else:
            if tree.word in document.bag_of_words:
                return self.classify(document, tree.children[0])
            else:
                return self.classify(document, tree.children[1])

    def get_tree(self, documents, vocabulary):
        """Returns the decision tree for the documents given the vocabulary;
        it is done recursively by passing the updated vocabulary at each call.

        Checks for the base cases first and returns a leaf node if they succeed;
        Otherwise, it finds the most informative word, splits the document and calls recursively for each child.
        """
        if self.contain_one_class(documents):
            return ClassTreeNode(self.contain_one_class(documents))
        elif len(vocabulary) == 0:
            return ClassTreeNode(self.get_majority_class(documents))
        else:
            most_informative_word = self.get_most_informative_word(documents, vocabulary)
            tree_node = WordTreeNode(most_informative_word)
            with_word, without_word = self.get_split_data(most_informative_word, documents)
            vocabulary.remove(most_informative_word)
            for subset in [with_word, without_word]:
                if len(subset) == 0:
                    tree_node.children.append(ClassTreeNode(self.get_majority_class(subset)))
                else:
                    tree_node.children.append(self.get_tree(subset, vocabulary))
            vocabulary.add(most_informative_word)
            return tree_node

    def contain_one_class(self, documents):
        """Returns None if the documents contain more than a class; returns the class otherwise."""
        classes = []
        for document in documents:
            if document.c not in classes:
                if len(classes) == 0:
                    classes.append(document.c)
                else:
                    return None
        if len(classes) == 1:
            return classes[0]
        else:
            return None

    def get_majority_class(self, documents):
        """Returns the majority class for the documents."""
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

    def get_split_data(self, word, documents):
        """Returns two lists given the documents and a word:
        the first contains documents where the word appears, the second contains documents without the word."""
        with_word = set([])
        without_word = set([])
        for document in documents:
            if word in document.bag_of_words:
                with_word.add(document)
            else:
                without_word.add(document)
        return with_word, without_word

    def get_most_informative_word(self, documents, vocabulary):
        """Returns the word in the given vocabulary with the highest information gain for the given documents."""
        most_informative_word = None
        most_informative_word_gain = 0
        for word in vocabulary:
            gain = self.get_information_gain(word, documents)
            if most_informative_word == None or gain >= most_informative_word_gain:
                most_informative_word = word
                most_informative_word_gain = gain
        return most_informative_word

    def get_information_gain(self, word, documents):
        """Returns the information gain of the given word for the documents;
        the subsets to subtract their entropies from the entropy of the original set
        are the sets with and without the word."""
        gain = self.get_entropy(documents)
        with_word, without_word = self.get_split_data(word, documents)
        gain -= self.get_entropy(with_word) * len(with_word) / len(documents)
        gain -= self.get_entropy(without_word) * len(without_word) / len(documents)
        return gain

    def get_entropy(self, documents):
        """Returns the entropy of the given documents;
        it is computed as: H(S) = - sum_{x \in X) p(x) * log(p(x), 2),
        where H(S) is the entropy of set S, x is a class in X and p(x) is the ratio of examples in S with class x."""
        entropy = 0
        for c in self.classes:
            count = 0
            for document in documents:
                if document.c == c:
                    count += 1
            if count != 0 and len(documents) != 0:
                ratio = count / len(documents)
                entropy -= ratio * log(ratio, 2)
        return entropy