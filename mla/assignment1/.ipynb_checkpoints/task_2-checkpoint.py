import math
import numpy as np
import matplotlib.pyplot as plt

class knn_mnist:
    def __init__(self, training_data, training_labels):
        self.training_data = training_data
        self.training_labels = training_labels

    def distance(self, x_1, x_2):
        return np.cumsum(math.sqrt((x_1 - x_2) ** 2))

    def predict(self, x, k):
        distances = np.array([self.distance(x, x_train) for x_train in self.training_data])
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = self.training_labels[k_nearest_indices]
        return np.argmax(np.bincount(k_nearest_labels))

    def evaluate(self, test_data, test_labels, k):
        predictions = np.array([self.predict(x, k) for x in test_data])
        accuracy = np.sum(predictions == test_labels) / len(test_labels)
        return accuracy

data_file_path = "MNIST-5-6-Subset/MNIST-5-6-Subset.txt"
data_matrix = np.loadtxt(data_file_path).reshape(1877, 784)

training_data = data_matrix[:1500]
training_labels = np.loadtxt("MNIST-5-6-Subset/MNIST-5-6-Subset-Train-Labels.txt")[:1500]

test_data = data_matrix[1500:2000]
test_labels = np.loadtxt("MNIST-5-6-Subset/MNIST-5-6-Subset-Test-Labels.txt")

knn = knn_mnist(training_data, training_labels)
accuracy = knn.evaluate(test_data, test_labels, 3)
print("Accuracy: ", accuracy)

