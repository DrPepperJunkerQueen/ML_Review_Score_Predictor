import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

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
y = df['rating'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# LOGIKA WCZYTYWANIA / NADPISYWANIA
# ==========================================
model_path = "milestone3_ridge_model.joblib"
vec_path = "milestone3_vectorizer.joblib"

istnieja_jakies_pliki = os.path.exists(model_path) or os.path.exists(vec_path)
wymus_nadpisanie = False

if istnieja_jakies_pliki:
    odp = input(f"\nZnaleziono zapisane pliki na dysku.\nCzy chcesz NADPISAĆ WSZYSTKO i trenować od nowa? (t/N - domyślnie 'N', wygeneruje tylko braki): ").strip().lower()
    if odp == 't':
        wymus_nadpisanie = True

# ==========================================
# KROK 3: WEKTORYZACJA
# ==========================================
if wymus_nadpisanie or not os.path.exists(vec_path):
    print("\nBudowanie słownika i zamiana tekstu na liczby (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    joblib.dump(vectorizer, vec_path)
    print("Zapisano wektoryzator.")
else:
    print("\nWczytywanie wektoryzatora z dysku...")
    vectorizer = joblib.load(vec_path)
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

# ==========================================
# KROK 4: BUDOWA I TRENING MODELU
# ==========================================
if wymus_nadpisanie or not os.path.exists(model_path):
    print("Trenowanie modelu (Ridge Regression)...")
    model = Ridge(alpha=1.0)
    model.fit(X_train_vec, y_train)
    joblib.dump(model, model_path)
    print("Zapisano model na dysku.")
else:
    print("Wczytywanie wytrenowanego modelu z dysku...")
    model = joblib.load(model_path)

# ==========================================
# KROK 5: PRZEWIDYWANIE I PRZYCINANIE DO SKALI
# ==========================================
y_pred_raw = model.predict(X_test_vec)
y_pred = np.clip(y_pred_raw, 1.0, 5.0)

# ==========================================
# KROK 6: METRYKI SKUTECZNOŚCI
# ==========================================
SKALA = 5.0 - 1.0  # rozpiętość skali = 4.0

bledy_bezwzgledne = np.abs(y_pred - y_test.values)
skutecznosci_jednostkowe = 1.0 - bledy_bezwzgledne / SKALA
ogolna_skutecznosc = skutecznosci_jednostkowe.mean()

mae = bledy_bezwzgledne.mean()
rmse = np.sqrt(((y_pred - y_test.values) ** 2).mean())

print(f"\n--- WYNIKI MODELU ---")
print(f"Ogólna skuteczność (Proximity Accuracy): {ogolna_skutecznosc * 100:.2f}%")
print(f"  (100% = idealne trafienie, 0% = maksymalny możliwy błąd = 4 gwiazdki)")
print(f"Średni błąd bezwzględny (MAE):           {mae:.4f} gwiazdki")
print(f"Pierwiastek błędu kwadratowego (RMSE):   {rmse:.4f} gwiazdki")

# ==========================================
# KROK 7: WYKRES — RZECZYWISTE vs PRZEWIDYWANE
# ==========================================
print("\nRysowanie wykresu rzeczywistych vs przewidywanych ocen...")
print("(zamknij okno wykresu, aby przejść dalej)")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.05, color='steelblue', s=5)
plt.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Idealne trafienie')
plt.xlabel("Rzeczywista ocena")
plt.ylabel("Przewidywana ocena")
plt.title("Rzeczywiste vs Przewidywane oceny")
plt.xticks([1, 2, 3, 4, 5])
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# KROK 8: ROZKŁAD BŁĘDÓW
# ==========================================
print("Rysowanie rozkładu błędów przewidywań...")
print("(zamknij okno wykresu, aby przejść dalej)")

plt.figure(figsize=(9, 5))
sns.histplot(bledy_bezwzgledne, bins=40, kde=True, color='coral')
plt.axvline(mae, color='darkred', linestyle='--', label=f'Średni błąd = {mae:.2f}')
plt.title("Rozkład błędów bezwzględnych przewidywań")
plt.xlabel("Błąd bezwzględny (gwiazdki)")
plt.ylabel("Liczba próbek")
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# KROK 9: TESTY NA ŻYWO
# ==========================================
print("\n--- TESTY NA PRZYKŁADOWYCH ZDANIACH ---")
test_zdania = [
    "I love this product, it works perfectly!",
    "It's okay, but the delivery was very slow.",
    "Completely useless. Broke after one day."
]
test_vec = vectorizer.transform(test_zdania)
przewidywania = np.clip(model.predict(test_vec), 1.0, 5.0)

for zdanie, ocena in zip(test_zdania, przewidywania):
    print(f"[{ocena:.2f} gwiazdki] -> {zdanie}")

# ==========================================
# KROK 10: TESTOWANIE NA DOWOLNYM PLIKU TEKSTOWYM
# ==========================================
print("\n" + "=" * 50)
print("TESTOWANIE NA WŁASNYM PLIKU (.txt)")
print("Wciśnij Enter bez wpisywania nazwy, aby zakończyć.")
print("=" * 50)

while True:
    nazwa_pliku = input("\nWpisz nazwę pliku .txt (np. recenzja.txt) lub naciśnij Enter, aby zakończyć: ")

    if not nazwa_pliku:
        print("Kończę program.")
        break

    if not os.path.exists(nazwa_pliku):
        print(f"BŁĄD: Nie znaleziono pliku '{nazwa_pliku}'. Upewnij się, że plik znajduje się w tym samym folderze co skrypt.")
        continue

    try:
        with open(nazwa_pliku, "r", encoding="utf-8") as f:
            tekst_z_pliku = f.read()

        if len(tekst_z_pliku.strip()) < 5:
            print("Plik jest pusty lub zawiera za mało tekstu.")
            continue

        wektor_pliku = vectorizer.transform([tekst_z_pliku])
        wynik_raw = model.predict(wektor_pliku)[0]
        wynik = float(np.clip(wynik_raw, 1.0, 5.0))

        print(f"\n--- WYNIK ANALIZY PLIKU '{nazwa_pliku}' ---")
        print(f"TREŚĆ (początek): {tekst_z_pliku[:150]}...")
        print(f"PRZEWIDYWANA OCENA: {wynik:.2f} GWIAZDKI/EK")

    except Exception as e:
        print(f"Wystąpił błąd podczas odczytu pliku: {e}")

print("\n--- KONIEC PROGRAMU ---")