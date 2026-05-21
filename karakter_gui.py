import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt

# harflerin piksel matrisleri (7 satir 5 sutun)
veri_seti = {
    "A": [
        "01110",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001"
    ],
    "B": [
        "11110",
        "10001",
        "10001",
        "11110",
        "10001",
        "10001",
        "11110"
    ],
    "C": [
        "01111",
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "01111"
    ],
    "D": [
        "11110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "11110"
    ],
    "E": [
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "11111"
    ]
}

harfler = ["A", "B", "C", "D", "E"]


# 7x5 matrisi 35 uzunluklu vektore ceviriyoruz
# 1 -> +1, 0 -> -1 yapiyoruz
def harfi_vektore_cevir(harf_matrisi):
    vektor = []
    for satir in harf_matrisi:
        for deger in satir:
            if deger == "1":
                vektor.append(1.0)
            else:
                vektor.append(-1.0)
    return np.array(vektor)


def cizimi_vektore_cevir(cizim_verisi):
    duz = cizim_verisi.reshape(35)
    return np.where(duz == 1, 1.0, -1.0)



def one_hot(index, sinif_sayisi):
    hedef = np.zeros(sinif_sayisi)
    hedef[index] = 1
    return hedef

X = np.array([harfi_vektore_cevir(veri_seti[h]) for h in harfler])
y = np.array([0, 1, 2, 3, 4])

def veri_artir(X, y):
    yeni_X = []
    yeni_y = []

    for x, etiket in zip(X, y):
        yeni_X.append(x.copy())
        yeni_y.append(etiket)

        siyah_indeksler = np.where(x == 1)[0]
        beyaz_indeksler = np.where(x == -1)[0]

        for indeks in siyah_indeksler:
            bozuk = x.copy()
            bozuk[indeks] = -1
            yeni_X.append(bozuk)
            yeni_y.append(etiket)

        for indeks in beyaz_indeksler[::3]:
            bozuk = x.copy()
            bozuk[indeks] = 1
            yeni_X.append(bozuk)
            yeni_y.append(etiket)

    return np.array(yeni_X), np.array(yeni_y)


X_artirilmis, y_artirilmis = veri_artir(X, y)


# ---------- Perceptron ----------
class PerceptronModel:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        self.lr = learning_rate
        self.W = np.zeros((input_size, output_size))
        self.b = np.zeros(output_size)
        self.hatalar = []

    def tahmin(self, x):
        skorlar = np.dot(x, self.W) + self.b
        return np.argmax(skorlar)

    def egit(self, X_train, y_train, epoch):
        self.hatalar = []
        for _ in range(epoch):
            hata_sayisi = 0
            for i in range(len(X_train)):
                x = X_train[i]
                gercek = y_train[i]
                tahmin = self.tahmin(x)

                if tahmin != gercek:
                    self.W[:, gercek] += self.lr * x
                    self.b[gercek] += self.lr
                    self.W[:, tahmin] -= self.lr * x
                    self.b[tahmin] -= self.lr
                    hata_sayisi += 1

            self.hatalar.append(hata_sayisi)
        return self.hatalar


# ---------- MLP ----------
class MLPModel:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.03):
        self.lr = learning_rate
        np.random.seed(7)

        
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros(hidden_size)

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros(output_size)

        self.hatalar = []

    def tanh(self, z):
        return np.tanh(z)

    def tanh_turev(self, a):
        return 1 - a ** 2

    def softmax(self, z):
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def ileri_yayilim(self, x):
        z1 = np.dot(x, self.W1) + self.b1
        a1 = self.tanh(z1)

        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.softmax(z2)

        return a1, a2

    def tahmin(self, x):
        a1, a2 = self.ileri_yayilim(x)
        return np.argmax(a2)

    def egit(self, X_train, y_train, epoch):
        self.hatalar = []

        for _ in range(epoch):
            siralama = np.arange(len(X_train))
            np.random.shuffle(siralama)  
            toplam_loss = 0

            for i in siralama:
                x = X_train[i]
                gercek = y_train[i]
                hedef = one_hot(gercek, len(harfler))

                a1, a2 = self.ileri_yayilim(x)

                loss = -np.sum(hedef * np.log(a2 + 1e-9))
                toplam_loss += loss

                dz2 = a2 - hedef
                dW2 = np.outer(a1, dz2)
                db2 = dz2

                dz1 = np.dot(self.W2, dz2) * self.tanh_turev(a1)
                dW1 = np.outer(x, dz1)
                db1 = dz1

                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

            ortalama_loss = toplam_loss / len(X_train)
            self.hatalar.append(ortalama_loss)

        return self.hatalar

model = None
model_tipi = None
hata_listesi = []


# ---------- Arayuz ----------
pencere = tk.Tk()
pencere.title("Karakter Tanima - Perceptron ve MLP")
pencere.geometry("920x680")
pencere.resizable(False, False)

baslik = tk.Label(pencere, text="KARAKTER TANIMA SİSTEMİ", font=("Arial", 18, "bold"))
baslik.pack(pady=12)

ana_frame = tk.Frame(pencere)
ana_frame.pack()

sol_frame = tk.Frame(ana_frame)
sol_frame.grid(row=0, column=0, padx=30, sticky="n")

sag_frame = tk.Frame(ana_frame)
sag_frame.grid(row=0, column=1, padx=30, sticky="n")


tk.Label(sol_frame, text="Model Sec:", font=("Arial", 10, "bold")).pack()

secili_model = tk.StringVar()
model_combo = ttk.Combobox(sol_frame, values=["MLP", "Perceptron"],
                            textvariable=secili_model, state="readonly", width=18)
model_combo.current(0)
model_combo.pack(pady=5)

tk.Label(sol_frame, text="Ogrenme Orani (LR):", font=("Arial", 10, "bold")).pack()
lr_entry = tk.Entry(sol_frame, width=18)
lr_entry.insert(0, "0.03")
lr_entry.pack(pady=5)

tk.Label(sol_frame, text="Epoch:", font=("Arial", 10, "bold")).pack()
epoch_entry = tk.Entry(sol_frame, width=18)
epoch_entry.insert(0, "1000")
epoch_entry.pack(pady=5)

sonuc_label = tk.Label(sol_frame, text="Sonuc: ---", font=("Arial", 11, "bold"),
                        wraplength=300, justify="center")
sonuc_label.pack(pady=15)


def modeli_egit():
    global model, model_tipi, hata_listesi

    try:
        lr = float(lr_entry.get())
        epoch = int(epoch_entry.get())
    except:
        messagebox.showerror("Hata", "Deger hatasi! Kontrol et.")
        return

    model_tipi = secili_model.get()

    if model_tipi == "Perceptron":
        model = PerceptronModel(input_size=35, output_size=5, learning_rate=lr)
        hata_listesi = model.egit(X, y, epoch)
    else:
        model = MLPModel(input_size=35, hidden_size=30, output_size=5, learning_rate=lr)
        hata_listesi = model.egit(X_artirilmis, y_artirilmis, epoch)

    sonuc_label.config(text=f"Sonuc: {model_tipi} egitildi.")
    messagebox.showinfo("Bilgi", f"{model_tipi} modeli egitildi!")


tk.Button(sol_frame, text="Modeli Egit", width=28, command=modeli_egit).pack(pady=6)

tk.Label(sol_frame, text="Test Edilecek Harf:", font=("Arial", 10, "bold")).pack(pady=6)

secili_harf = tk.StringVar()
harf_combo = ttk.Combobox(sol_frame, values=harfler, textvariable=secili_harf,
                           state="readonly", width=18)
harf_combo.current(0)
harf_combo.pack()


def data_uzerinden_tahmin():
    if model is None:
        messagebox.showwarning("Uyari", "Once modeli egit!")
        return

    secilen_harf = secili_harf.get()
    x = harfi_vektore_cevir(veri_seti[secilen_harf])
    tahmin_index = model.tahmin(x)
    tahmin_harf = harfler[tahmin_index]

    sonuc_label.config(text=f"Model: {model_tipi}\nSecilen: {secilen_harf}\nTahmin: {tahmin_harf}")


tk.Button(sol_frame, text="Data Uzerinden Tahmin Et", width=28,
          command=data_uzerinden_tahmin).pack(pady=6)


def cizimden_tahmin_et():
    if model is None:
        messagebox.showwarning("Uyari", "Once modeli egit!")
        return

    x = cizimi_vektore_cevir(cizim_verisi)
    tahmin_index = model.tahmin(x)
    tahmin_harf = harfler[tahmin_index]

    sonuc_label.config(text=f"Model: {model_tipi}\nCizim Tahmini: {tahmin_harf}")


def grafik_goster():
    if model is None or len(hata_listesi) == 0:
        messagebox.showwarning("Uyari", "Once modeli egit!")
        return

    plt.figure("Egitim Grafigi")
    plt.plot(hata_listesi)
    plt.xlabel("Epoch")

    if model_tipi == "Perceptron":
        plt.ylabel("Hata Sayisi")
        plt.title("Perceptron Hata Grafigi")
    else:
        plt.ylabel("Loss")
        plt.title("MLP Loss Grafigi")

    plt.grid(True)
    plt.show()


tk.Button(sol_frame, text="Egitim Grafigini Goster", width=28,
          command=grafik_goster).pack(pady=6)


def accuracy_hesapla():
    if model is None:
        messagebox.showwarning("Uyari", "Once modeli egit!")
        return

    dogru = 0
    for i in range(len(X)):
        if model.tahmin(X[i]) == y[i]:
            dogru += 1

    acc = dogru / len(X)
    sonuc_label.config(text=f"Model: {model_tipi}\nBasari: %{acc * 100:.2f}")


tk.Button(sol_frame, text="Basari Oranini Goster", width=28,
          command=accuracy_hesapla).pack(pady=6)


tk.Label(sag_frame, text="CIZIM ALANI", font=("Arial", 11, "bold")).pack(pady=6)

hucreler = []
cizim_verisi = np.zeros((7, 5), dtype=int)


def hucre_degistir(i, j):
    if cizim_verisi[i][j] == 0:
        cizim_verisi[i][j] = 1
        hucreler[i][j].config(bg="black")
    else:
        cizim_verisi[i][j] = 0
        hucreler[i][j].config(bg="white")


grid_frame = tk.Frame(sag_frame)
grid_frame.pack()

for i in range(7):
    satir = []
    for j in range(5):
        btn = tk.Button(grid_frame, width=5, height=2, bg="white",
                        command=lambda i=i, j=j: hucre_degistir(i, j))
        btn.grid(row=i, column=j, padx=1, pady=1)
        satir.append(btn)
    hucreler.append(satir)


def secili_harfi_cizime_yukle():
    secilen_harf = secili_harf.get()
    matris = veri_seti[secilen_harf]

    for i in range(7):
        for j in range(5):
            deger = int(matris[i][j])
            cizim_verisi[i][j] = deger
            hucreler[i][j].config(bg="black" if deger == 1 else "white")

    sonuc_label.config(text=f"Sonuc: {secilen_harf} yuklendi.")


def cizimi_temizle():
    for i in range(7):
        for j in range(5):
            cizim_verisi[i][j] = 0
            hucreler[i][j].config(bg="white")
    sonuc_label.config(text="Sonuc: ---")


tk.Button(sag_frame, text="Secilen Harfi Cizim Alanina Yukle", width=38,
          command=secili_harfi_cizime_yukle).pack(pady=8)

tk.Button(sag_frame, text="Cizim Alanindan Tahmin Et", width=38,
          command=cizimden_tahmin_et).pack(pady=4)

tk.Button(sag_frame, text="Cizimi Temizle", width=38,
          command=cizimi_temizle).pack(pady=4)

tk.Label(pencere,
         text="Not: Perceptron ve MLP modelleri var. Girdi 35 boyutlu vektor.",
         font=("Arial", 9)).pack(pady=15)

pencere.mainloop()