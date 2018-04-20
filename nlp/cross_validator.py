from math import floor
from nlp.naive_bayes import NaiveBayes
from nlp.utilities import ConfusionMatrix
from nlp.knn import KNN
from nlp.id3 import ID3
from abc import ABC, abstractmethod
from nlp.dataset import Dataset
import sys
from datetime import datetime


class CrossValidator(ABC):

    """An abstract class to perform K-Fold Cross-Validation on a set with a classifier."""

    def __init__(self, dataset, folds):
        """Initializes the tool given the dataset and the number of folds."""
        self.dataset = dataset
        self.folds = folds
        self.sets = self.get_sets()

    def get_sets(self):
        """Returns as many balanced lists of documents as many folds."""
        documents_by_class = Dataset.get_documents_by_class(self.dataset.documents)
        sets = []
        for i in range(0, self.folds):
            set = []
            for c in documents_by_class:
                n = floor(len(documents_by_class[c]) / self.folds)
                for j in range(0, n):
                    set.append(documents_by_class[c][i * n + j])
            sets.append(set)
        return sets

    def get_training_testing_sets(self, n):
        """Returns the set at position n as the testing set and merges the others and returns them as training set."""
        training = []
        testing = []
        for i in range(0, self.folds):
            if i == n:
                testing += self.sets[i]
            else:
                training += self.sets[i]
        return training, testing

    def cross_validate(self):
        """Performs cross validation; prints confusion matrix and accuracy at the end of each fold validation."""
        start = datetime.now()
        confusion_matrices = []
        for i in range(0, self.folds):
            print("Cross-Validating (" + str(i + 1) + " of " + str(self.folds) + ")")
            training, testing = self.get_training_testing_sets(i)
            confusion_matrix = self.get_test_results(training, testing)
            confusion_matrices.append(confusion_matrix)
            print(str(confusion_matrix) + "\n")
        sum = 0
        for confusion_matrix in confusion_matrices:
            sum += confusion_matrix.get_accuracy()
        avg = sum / len(confusion_matrices)
        print("Average accuracy is " + str(round(100 * avg, 2)) + "%.\n")
        finish = datetime.now()
        delta = finish - start
        print(str(delta))

    @abstractmethod
    def get_test_results(self, training, testing):
        """(Abstract) Returns the confusion matrix from the fold validation for the given training and testing sets."""
        return None

class NaiveBayesCrossValidator(CrossValidator):

    """A K-Fold Cross-Validator for the Naive Bayes classifier."""

    def get_test_results(self, training, testing):
        """Returns the confusion matrix from the fold validation for the given training and testing sets."""
        nb = NaiveBayes(training)
        confusion_matrix = ConfusionMatrix(nb.classes)
        for document in testing:
            predicted = nb.get_prediction(document)
            actual = document.c
            confusion_matrix.add(actual, predicted)
        return confusion_matrix

class KNNCrossValidator(CrossValidator):

    """A K-Fold Cross-Validator for the KNN classifier."""

    def __init__(self, dataset, folds, k):
        """Initializes the KNN K-Fold Cross-Validator with the dataset, the folds and the number of neighbours."""
        CrossValidator.__init__(self, dataset, folds)
        self.k = k

    def get_test_results(self, training, testing):
        """Returns the confusion matrix from the fold validation for the given training and testing sets."""
        knn = KNN(training, self.k)
        confusion_matrix = ConfusionMatrix(knn.classes)
        for document in testing:
            predicted = knn.get_prediction(document)
            actual = document.c
            confusion_matrix.add(actual, predicted)
        return confusion_matrix

class ID3CrossValidator(CrossValidator):

    """A K-Fold Cross-Validator for the ID3 classifier."""

    def get_test_results(self, training, testing):
        """Returns the confusion matrix from the fold validation for the given training and testing sets."""
        id3 = ID3(training)
        confusion_matrix = ConfusionMatrix(id3.classes)
        for document in testing:
            predicted = id3.get_prediction(document)
            actual = document.c
            confusion_matrix.add(actual, predicted)
        return confusion_matrix


# cross_validator.py <folds> <dataset> <classifier> <options>

if __name__ == '__main__':
    print(" ")
    format = "cross_validator.py <folds> <dataset> <classifier> [<options>]\n"
    arguments = sys.argv
    if len(arguments) >= 4:
        folds = 0
        try:
            folds = int(arguments[1])
        except:
            print("Invalid format for number of folds (should be integer).")
            print("\n  >  " + format)
            exit()
        dataset = Dataset.get_dataset(arguments[2])
        if dataset is not None:
            cv = None
            if arguments[3] == "naive_bayes":
                cv = NaiveBayesCrossValidator(dataset, folds)
            elif arguments[3] == "knn":
                if len(arguments) == 5:
                    k = 0
                    try:
                        k = int(arguments[4])
                    except:
                        print("Invalid number format for number of neighbours")
                        print("\n  >  cross_validator.py <folds> <dataset> <classifier> <neighbours>\n")
                        exit()
                    cv = KNNCrossValidator(dataset, folds, k)
                else:
                    print("Invalid number of arguments, missing number of neighbours.")
                    print("\n  >  cross_validator.py <folds> <dataset> <classifier> <neighbours>\n")
                    exit()
            elif arguments[3] == "id3":
                cv = ID3CrossValidator(dataset, folds)
            else:
                print("Invalid classifier name (should be \"naive_bayes\", \"knn\" or \"id3\").")
                print("\n  >  " + format)
                exit()
            cv.cross_validate()
        else:
            print("Invalid dataset name.")
            print("\n  >  " + format)
            exit()
    else:
        print("Invalid number of arguments.")
        print("\n  >  " + format)
        exit()

























