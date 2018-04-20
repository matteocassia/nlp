from random import uniform, randint
from math import pow, sqrt


class KMeans:

    def __init__(self, word_vectors, k):
        self.word_vectors = word_vectors
        self.dimensions = len(word_vectors[0][1])
        self.boundaries = self.get_boundaries(word_vectors)
        self.k = k
        self.cluster(word_vectors)

    def cluster(self, word_vectors):
        clusters = self.get_initial_clusters()
        while True:
            for word_vector in word_vectors:
                vector = word_vector[1]
                best_centroid = None
                best_distance = float("+inf")
                for centroid in clusters:
                    distance = self.get_euclidean_distance(vector, centroid)
                    if distance < best_distance:
                        best_distance = distance
                        best_centroid = centroid
                clusters[best_centroid].append(word_vector)
            self.print_clusters(clusters)
            new_clusters = self.get_new_clusters(clusters)
            for i in range(0, self.k - len(new_clusters)):
                point = self.get_random_point()
                new_clusters[point] = []
            if new_clusters.keys() == clusters.keys():
                break
            else:
                clusters = new_clusters

    def print_clusters(self, clusters):
        i = 1
        for centroid in clusters:
            row = str(i) + "\n\n"
            i += 1
            for word_vector in clusters[centroid]:
                row += word_vector[0] + ","
            print(row)

    def get_new_clusters(self, clusters):
        new_clusters = {}
        for centroid in clusters:
            new_centroid = []
            for n in range(0, self.dimensions):
                sum = 0
                for word_vector in clusters[centroid]:
                    sum += word_vector[1][n]
                if len(clusters[centroid]) != 0:
                    new_centroid.append(sum / len(clusters[centroid]))
            if len(new_centroid) != 0:
                new_clusters[tuple(new_centroid)] = []
        return new_clusters

    def get_initial_clusters(self):
        clusters = {}
        for i in range(0, self.k):
            point = self.get_random_point()
            clusters[point] = []
        return clusters

    def get_euclidean_distance(self, vector_a, vector_b):
        sum = 0
        for n in range(0, self.dimensions):
            sum += pow(vector_a[n] - vector_b[n], 2)
        return sqrt(sum)

    def get_manhattan_distance(self, vector_a, vector_b):
        sum = 0
        for n in range(0, self.dimensions):
            sum += abs(vector_a[n] - vector_b[n])
        return sum

    def get_random_point(self):
        point = []
        for n in range(0, self.dimensions):
            minimum = self.boundaries[n][0]
            maximum = self.boundaries[n][1]
            point.append(self.get_random_number(minimum, maximum))
        return tuple(point)

    def get_random_number(self, minimum, maximum):
        return uniform(minimum, maximum)

    def get_boundaries(self, word_vectors):
        boundaries = []
        for n in range(0, self.dimensions):
            minimum = float("+inf")
            maximum = float("-inf")
            for word_vector in word_vectors:
                vector = word_vector[1]
                minimum = min(minimum, vector[n])
                maximum = max(maximum, vector[n])
            boundaries.append((minimum, maximum))
        return boundaries