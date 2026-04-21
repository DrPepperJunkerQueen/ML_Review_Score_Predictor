import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ustawienie stylu wykresów
sns.set_theme(style="whitegrid")

# ==========================================
# KROK 1: ŁADOWANIE DANYCH
# ==========================================
print("Ładowanie zbalansowanych danych...")
if not os.path.exists("zbalansowane_gotowe_do_treningu.parquet"):
    print("BŁĄD: Nie znaleziono pliku 'zbalansowane_gotowe_do_treningu.parquet'!")
    print("Uruchom najpierw Milestone 2, aby przygotować dane.")
    exit()

df = pd.read_parquet("zbalansowane_gotowe_do_treningu.parquet")

# ==========================================
# KROK 2: PODZIAŁ NA ZBIÓR TRENINGOWY I TESTOWY
# ==========================================
print("Dzielenie danych na zbiór do nauki (80%) i do testów (20%)...")
X = df['full_review']
y = df['rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# KROK 3: WEKTORYZACJA (Zamiana słów na liczby)
# ==========================================
print("Budowanie słownika i zamiana tekstu na liczby (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==========================================
# KROK 4: BUDOWA I TRENING MODELU
# ==========================================
print("Trenowanie modelu (Regresja Logistyczna)...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# ==========================================
# KROK 5: EWALUACJA (Egzamin modelu)
# ==========================================
y_pred = model.predict(X_test_vec)
skutecznosc = accuracy_score(y_test, y_pred)
print(f"\nOGÓLNA SKUTECZNOŚĆ MODELU: {skutecznosc * 100:.2f}%")

# ==========================================
# KROK 6: MACIERZ POMYŁEK (Wizualizacja)
# ==========================================
print("\nRysowanie Macierzy Pomyłek (zamknij okno wykresu, aby przejść dalej)...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5])
plt.title("Macierz Pomyłek (Confusion Matrix)")
plt.xlabel("Przewidywania modelu")
plt.ylabel("Rzeczywista ocena")
plt.show()

# ==========================================
# KROK 7: TESTY NA ŻYWO (Zdefiniowane w kodzie)
# ==========================================
print("\n--- TESTY NA PRZYKŁADOWYCH ZDANIACH ---")
test_zdania = [
    "I love this product, it works perfectly!",
    "It's okay, but the delivery was very slow.",
    "Completely useless. Broke after one day."
]
test_vec = vectorizer.transform(test_zdania)
przewidywania = model.predict(test_vec)

for zdanie, ocena in zip(test_zdania, przewidywania):
    print(f"[{ocena} gwiazdki] -> {zdanie}")

# ==========================================
# KROK 8: TESTOWANIE NA DOWOLNYM PLIKU TEKSTOWYM
# ==========================================
print("\n" + "=" * 50)
print("TESTOWANIE NA WŁASNYM PLIKU (.txt)")
print("=" * 50)

nazwa_pliku = input("Wpisz nazwę pliku .txt (np. recenzja.txt) lub naciśnij Enter, aby zakończyć: ")

if nazwa_pliku:
    if os.path.exists(nazwa_pliku):
        try:
            with open(nazwa_pliku, "r", encoding="utf-8") as f:
                tekst_z_pliku = f.read()

            if len(tekst_z_pliku.strip()) < 5:
                print("Plik jest pusty lub zawiera za mało tekstu.")
            else:
                # Zamiana tekstu z pliku na wektor
                wektor_pliku = vectorizer.transform([tekst_z_pliku])
                wynik = model.predict(wektor_pliku)[0]

                print(f"\n--- WYNIK ANALIZY PLIKU '{nazwa_pliku}' ---")
                print(f"TREŚĆ (początek): {tekst_z_pliku[:150]}...")
                print(f"PRZEWIDYWANA OCENA: {wynik} GWIAZDKI/EK")

        except Exception as e:
            print(f"Wystąpił błąd podczas odczytu pliku: {e}")
    else:
        print(
            f"BŁĄD: Nie znaleziono pliku o nazwie '{nazwa_pliku}'. Upewnij się, że plik znajduje się w tym samym folderze co skrypt.")

print("\n--- KONIEC PROGRAMU ---")