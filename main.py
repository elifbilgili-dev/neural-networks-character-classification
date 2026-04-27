from PIL import Image
import numpy as np
import os

from perceptron import Perceptron
from mlp import MLP


# =========================
# 1. VERİ SETİNİ OKUMA
# =========================

dataset_klasoru = "dataset"
siniflar = ["A", "B", "C"]

X = []
y = []

for etiket, harf in enumerate(siniflar):
    harf_klasoru = os.path.join(dataset_klasoru, harf)

    for dosya_adi in os.listdir(harf_klasoru):
        if dosya_adi.endswith(".png"):
            dosya_yolu = os.path.join(harf_klasoru, dosya_adi)

            resim = Image.open(dosya_yolu).convert("L")
            resim = resim.resize((28, 28))

            piksel = np.array(resim)
            piksel = piksel / 255.0

            vektor = piksel.flatten()

            X.append(vektor)
            y.append(etiket)

X = np.array(X)
y = np.array(y)

print("Veri seti okundu.")
print("X boyutu:", X.shape)
print("y boyutu:", y.shape)


# =========================
# 2. VERİYİ KARIŞTIRMA
# =========================

np.random.seed(42)
indeksler = np.arange(len(X))
np.random.shuffle(indeksler)

X = X[indeksler]
y = y[indeksler]


# =========================
# 3. EĞİTİM VE TEST AYIRMA
# =========================

egitim_sayisi = 6

X_train = X[:egitim_sayisi]
y_train = y[:egitim_sayisi]

X_test = X[egitim_sayisi:]
y_test = y[egitim_sayisi:]

print("\nEğitim veri sayısı:", len(X_train))
print("Test veri sayısı:", len(X_test))


# =========================
# 4. PERCEPTRON MODELİ
# =========================

perceptron = Perceptron(
    input_size=784,
    class_count=3,
    learning_rate=0.1,
    epochs=100
)

print("\nPerceptron eğitimi başlıyor...")
perceptron.train(X_train, y_train)

train_acc = perceptron.accuracy(X_train, y_train)
test_acc = perceptron.accuracy(X_test, y_test)

print("\n--- Perceptron Sonuçları ---")
print("Eğitim başarısı:", train_acc)
print("Test başarısı:", test_acc)

print("\nTest gerçek etiketler:", y_test)
print("Test tahminleri:", perceptron.predict(X_test))

# =========================
# 5. MLP MODELİ
# =========================

mlp = MLP(
    input_size=784,
    hidden_size=16,
    output_size=3,
    learning_rate=0.1,
    epochs=100
)

print("\nMLP eğitimi başlıyor...")
mlp.train(X_train, y_train)

mlp_train_acc = mlp.accuracy(X_train, y_train)
mlp_test_acc = mlp.accuracy(X_test, y_test)

print("\n--- MLP Sonuçları ---")
print("Eğitim başarısı:", mlp_train_acc)
print("Test başarısı:", mlp_test_acc)

print("\nTest gerçek etiketler:", y_test)
print("MLP test tahminleri:", mlp.predict(X_test))