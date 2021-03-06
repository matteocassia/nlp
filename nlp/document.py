from re import sub

stopwords = ["the", "and", "to", "a", "an", "i", "it", "this", "of", "for", "my", "on", "with", "you", "that", "in", "have"]

class Document:

    """A data structure to parse and represent a text document as a bag of words."""

    def __init__(self, text, preserve_duplicates=False, remove_stopwords=False, ngrams=None):
        """Initializes a document object given the raw text and a class;
        options include the possibility of preserving duplicate words in the document,
        removing the stopwords and adding ngrams (as a list of the n values)."""
        self.text = text
        bag_of_words = Document.get_bag_of_words(text)
        if remove_stopwords == True:
            bag_of_words = Document.get_bag_without_stopwords(bag_of_words)
        if ngrams != None and type(ngrams) is list:
            bag_of_words += Document.get_ngrams(ngrams, bag_of_words)
        if preserve_duplicates == False:
            bag_of_words = set(bag_of_words)
        self.bag_of_words = bag_of_words

    def __str__(self):
        """Returns the string representation of the document as a class-text pair."""
        return str(self.c) + " " + str(self.text)

    @staticmethod
    def get_bag_without_stopwords(bag_of_words):
        """Updates the bag of words by removing the stopwords."""
        new_bag = []
        for word in bag_of_words:
            if word not in stopwords:
                new_bag.append(word)
        return new_bag

    @staticmethod
    def get_ngrams(ns, bag_of_words):
        """Returns the ngrams given the bag of words and the desired ngram size."""
        ngrams = []
        for n in ns:
            number_of_ngrams = len(bag_of_words) - n + 1
            for i in range(0, number_of_ngrams):
                ngram = ""
                for j in range(0, n):
                    ngram += bag_of_words[i + j]
                    if j != n - 1:
                        ngram += " "
                ngrams.append(ngram)
        return ngrams

    @staticmethod
    def get_bag_of_words(text):
        """Returns the bag of words as a list of individual words;
        it is generated by splitting according to a regex."""
        parsed_text = Document.get_parsed_text(text)
        bag_of_words = parsed_text.split(" ")
        return bag_of_words

    @staticmethod
    def get_parsed_text(text):
        """Returns the given text parsed according to the regex and methods."""
        return sub(r"[^A-Za-z\']+", ' ', text).strip().lower()

class ClassDocument(Document):

    def __init__(self, text, c, preserve_duplicates=False, remove_stopwords=False, ngrams=None):
        Document.__init__(self, text, preserve_duplicates=preserve_duplicates, remove_stopwords=remove_stopwords,
                          ngrams=ngrams)
        self.c = c