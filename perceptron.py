
import numpy as np


class Perceptron:
    def __init__(self, input_size, class_count, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.class_count = class_count

        # Ağırlıklar ve bias başlangıçta küçük rastgele değerler alır
        self.weights = np.random.randn(class_count, input_size) * 0.01
        self.bias = np.zeros(class_count)

    def predict_one(self, x):
        # Her sınıf için skor hesaplanır
        scores = np.dot(self.weights, x) + self.bias

        # En yüksek skora sahip sınıf tahmin edilir
        return np.argmax(scores)

    def predict(self, X):
        predictions = []

        for x in X:
            prediction = self.predict_one(x)
            predictions.append(prediction)

        return np.array(predictions)

    def train(self, X, y):
        for epoch in range(self.epochs):
            errors = 0

            for i in range(len(X)):
                x = X[i]
                true_label = y[i]

                predicted_label = self.predict_one(x)

                # Eğer tahmin yanlışsa ağırlıkları güncelle
                if predicted_label != true_label:
                    self.weights[true_label] += self.learning_rate * x
                    self.bias[true_label] += self.learning_rate

                    self.weights[predicted_label] -= self.learning_rate * x
                    self.bias[predicted_label] -= self.learning_rate

                    errors += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs} - Hata sayısı: {errors}")

    def accuracy(self, X, y):
        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        return correct / len(y)