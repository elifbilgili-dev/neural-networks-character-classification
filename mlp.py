
import numpy as np


class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=500):
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Ağırlıklar küçük rastgele değerlerle başlatılır
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def one_hot(self, y, class_count):
        sonuc = np.zeros((len(y), class_count))
        for i in range(len(y)):
            sonuc[i][y[i]] = 1
        return sonuc

    def train(self, X, y):
        y_encoded = self.one_hot(y, 3)

        for epoch in range(self.epochs):
            # İleri yayılım
            hidden_input = np.dot(X, self.W1) + self.b1
            hidden_output = self.sigmoid(hidden_input)

            final_input = np.dot(hidden_output, self.W2) + self.b2
            final_output = self.sigmoid(final_input)

            # Hata hesaplama
            error = y_encoded - final_output

            # Geri yayılım
            output_delta = error * self.sigmoid_derivative(final_output)

            hidden_error = np.dot(output_delta, self.W2.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

            # Ağırlık güncelleme
            self.W2 += self.learning_rate * np.dot(hidden_output.T, output_delta)
            self.b2 += self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)

            self.W1 += self.learning_rate * np.dot(X.T, hidden_delta)
            self.b1 += self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

            if (epoch + 1) % 50 == 0:
                loss = np.mean(error ** 2)
                print(f"Epoch {epoch + 1}/{self.epochs} - Hata: {loss:.4f}")

    def predict(self, X):
        hidden_input = np.dot(X, self.W1) + self.b1
        hidden_output = self.sigmoid(hidden_input)

        final_input = np.dot(hidden_output, self.W2) + self.b2
        final_output = self.sigmoid(final_input)

        return np.argmax(final_output, axis=1)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        return correct / len(y)