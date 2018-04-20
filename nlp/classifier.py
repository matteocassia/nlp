class Classifier:

    """An abstract class to handle common functionalities to most classifiers."""

    def __init__(self, documents):
        """Initializes the classifier given the documents;
        it extracts the classes from the documents and stores them in a list."""
        self.documents = documents
        self.classes = self.get_classes()

    def get_classes(self):
        """Returns a list containing the unique classes found in the documents."""
        classes = []
        for document in self.documents:
            if document.c not in classes:
                classes.append(document.c)
        return classes

    def get_prediction(self, document):
        """Returns the predicted class for the given document."""
        return None