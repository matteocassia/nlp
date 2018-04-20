from nlp.document import ClassDocument
from json import loads
import random
from os import listdir


class Dataset:

    """A data structure to store and handle a group of documents."""

    def __init__(self, documents):
        """Initializes the dataset given the documents;
        if the normalisation option is set to true, the documents are randomly downsampled to be balanced."""
        self.documents = documents

    @staticmethod
    def get_minimum_class_count(documents_by_class):
        """Returns the size of the smallest class subset of the documents."""
        minimum_class_count = None
        for c in documents_by_class:
            if minimum_class_count == None:
                minimum_class_count = len(documents_by_class[c])
            else:
                minimum_class_count = min(minimum_class_count, len(documents_by_class[c]))
        return minimum_class_count

    @staticmethod
    def get_documents_by_class(documents):
        """Returns a dictionary for the documents by their class of the form (class:[documents for that class])."""
        documents_by_class = {}
        for document in documents:
            if document.c not in documents_by_class:
                documents_by_class[document.c] = []
            documents_by_class[document.c].append(document)
        return documents_by_class

    @staticmethod
    def get_split_documents(documents, split):
        """Returns a list with 100*(split)% documents for training and 100*(1-split)% documents for testing;
        both sets are balanced."""
        training = []
        testing = []
        documents_by_class = Dataset.get_documents_by_class(documents)
        for c in documents_by_class:
            n = int(len(documents_by_class[c]) * split)
            training += documents_by_class[c][0:n]
            testing += documents_by_class[c][n:]
        return training, testing

    @staticmethod
    def get_amazon_dataset(dataset_name, preserve_duplicates=False, remove_stopwords=False, balanced=True, ngrams=None):
        """Returns the dataset containing the negative and positive Amazon product reviews given the name of the set;
        options include preserving duplicate words, removing stopwords, the size of the ngrams as a list and
        balancing the dataset.

        Only reviews with score equal to 1 and 5 are kept and mapped as negative and positive, respectively."""
        documents = []
        map = {1.0: "-", 5.0: "+"}
        with open("documents/" + dataset_name) as file:
            lines = file.readlines()
            for line in lines:
                data = loads(line)
                score = data["overall"]
                if score in map:
                    rating = map[score]
                    text = data["reviewText"]
                    document = ClassDocument(text, rating, preserve_duplicates=preserve_duplicates,
                                        remove_stopwords=remove_stopwords, ngrams=ngrams)
                    documents.append(document)
        random.Random(42).shuffle(documents)
        dataset = ClassDataset(documents, normalise=balanced)
        return dataset

    @staticmethod
    def get_bbc_dataset():
        """Returns the dataset containing the articles found in the BBC dataset; the classes are the categories."""
        origin = "documents/bbc/"
        folders = ["business", "entertainment", "politics", "sport", "tech"]
        documents = []
        for folder in folders:
            path = origin + folder + "/"
            for filename in listdir(path):
                if filename.count(".txt") != 0:
                    filepath = path + filename
                    f = open(filepath, "r", encoding="utf-8")
                    t = f.read()
                    document = ClassDocument(t, folder)
                    documents.append(document)
        random.Random(42).shuffle(documents)
        return ClassDataset(documents, normalise=False)

    @staticmethod
    def get_dataset(name):
        """Returns the dataset matching the name provided (if any, otherwise None)."""
        if name in ["musical_instruments", "automotive", "instant_video", "beauty", "tools"]:
            return Dataset.get_amazon_dataset(name + ".json")
        elif name == "all_amazon":
            instruments = Dataset.get_amazon_dataset("musical_instruments.json", balanced=False).documents
            automotive = Dataset.get_amazon_dataset("automotive.json", balanced=False).documents
            instant_video = Dataset.get_amazon_dataset("instant_video.json", balanced=False).documents
            tools = Dataset.get_amazon_dataset("tools.json", balanced=False).documents
            beauty = Dataset.get_amazon_dataset("beauty.json", balanced=False).documents
            documents = instruments + automotive + instant_video + tools + beauty
            random.shuffle(documents)
            return Dataset(documents)
        elif name == "bbc":
            return Dataset.get_bbc_dataset()
        else:
            return None


class ClassDataset(Dataset):

    """An extension of the Dataset class meant to store ClassDocuments."""

    def __init__(self, documents, normalise=True):
        """Initialises the dataset with the documents and downsamples them if specified by the optional parameter."""
        if normalise:
            documents = ClassDataset.get_normalised_documents(documents)
        Dataset.__init__(self, documents)

    @staticmethod
    def get_normalised_documents(documents):
        """Returns the downsampled list of documents in order to be balanced;
        it is done by finding the size of the smallest class subset and downsizing each set to that size."""
        documents_by_class = ClassDataset.get_documents_by_class(documents)
        minimum_class_count = ClassDataset.get_minimum_class_count(documents_by_class)
        documents = []
        for c in documents_by_class:
            documents += documents_by_class[c][0:minimum_class_count]
        return documents


