class ConfusionMatrix:

    """A class to manage the results of a classification process."""

    def __init__(self, classes):
        """Initializes the confusion matrix given the classes;
        it generates a square matrix with as many rows and columns as classes filled with zeroes
        where the nth column and row are the true and predicted entries of the nth class."""
        self.classes = classes
        self.matrix = self.get_initial_matrix()

    def get_initial_matrix(self):
        """Returns a square matrix with as many rows and columns as classes filled with zeroes."""
        matrix = []
        for r in self.classes:
            row = []
            for c in self.classes:
                row.append(0)
            matrix.append(row)
        return matrix

    def add(self, actual, predicted):
        """Increases by one the value of the entry for the given true and predicted entries."""
        self.matrix[self.classes.index(predicted)][self.classes.index(actual)] += 1

    def get_accuracy(self):
        """Returns the accuracy as the ratio of the sum of the trace and the sum of every value in the matrix."""
        count = 0
        total = 0
        for i in range(0, len(self.classes)):
            for j in range(0, len(self.classes)):
                if i == j:
                    count += self.matrix[i][j]
                total += self.matrix[i][j]
        if count == 0:
            return 0.00
        return float(float(count) / float(total))

    def __str__(self):
        """Returns the string representation of the matrix, along with a statement on accuracy."""
        k = 18
        string = "|".ljust(k)
        for c in self.classes:
            string += str("| " + c).ljust(k)
        string += "\n"
        for i in range(0, len(self.classes)):
            string += str("| " + self.classes[i]).ljust(k)
            for j in range(0, len(self.classes)):
                string += str("| " + str(self.matrix[i][j])).ljust(k)
            string += "\n"
        string += "\nAccuracy is " + str(round(100 * self.get_accuracy(), 2)) + "%"
        return string

class Process:

    """A class to show the advancement of a time consuming procedure; it prints dots inline to show progress.

    e.g.

        p = Process("Generating vocabulary", 100, 2)
        > Generating vocabulary
        p.add()
        p.add()
        > Generating vocabulary.
        p.add()
        p.add()
        > Generating vocabulary..
    """

    def __init__(self, name, maximum, step):
        """Initializes the tool with the name of the process, the total number of steps and the percent step."""
        self.name = name
        self.maximum = maximum
        self.step = int(round(step * self.maximum / 100.00, 0)) + 1
        self.at = 0
        print(name, end="", flush=True)

    def add(self):
        """Increases the number of steps and if the total is a dividend of the percent step, it prints the update."""
        self.at += 1
        if self.at % self.step == 0:
            print(".", end="", flush=True)

    def finish(self):
        """Terminates the process and pirnts the final message."""
        print("done!\n")


