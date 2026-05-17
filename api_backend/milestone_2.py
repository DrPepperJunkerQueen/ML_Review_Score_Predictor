import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

# Ustawienie ładniejszego stylu wykresów
sns.set_theme(style="whitegrid")

PLIK_WEJSCIOWY = "przetworzona_probka_500k.parquet"
PLIK_WYJSCIOWY = "zbalansowane_gotowe_do_treningu.parquet"

# ==========================================
# SPRAWDZENIE ZALEŻNOŚCI
# ==========================================
if not os.path.exists(PLIK_WEJSCIOWY):
    print(f"BŁĄD: Nie znaleziono pliku '{PLIK_WEJSCIOWY}'!")
    print("Musisz najpierw uruchomić skrypt z Kamienia 1, aby pobrać dane.")
    exit()

print(f"Znaleziono plik '{PLIK_WEJSCIOWY}'. Ładowanie danych...")
df = pd.read_parquet(PLIK_WEJSCIOWY)

# ==========================================
# KROK 1: Czyszczenie zanieczyszczeń HTML
# ==========================================
print("Czyszczenie tagów HTML z tekstów...")
df['full_review'] = df['full_review'].str.replace(r'<[^>]+>', ' ', regex=True)
df['title'] = df['title'].str.replace(r'<[^>]+>', ' ', regex=True)
df['full_review'] = df['full_review'].str.replace(r'\s+', ' ', regex=True).str.strip()

# ==========================================
# KROK 2: Wykres początkowy
# ==========================================
print("\nRysowanie wykresu początkowego rozkładu ocen (ZAMKNIJ OKNO WYKRESU, ABY KOD SZEDŁ DALEJ)...")
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='rating', palette='viridis')
plt.title("Rozkład ocen w oryginalnej próbce (przed zbalansowaniem)")
plt.xlabel("Ocena (Gwiazdki)")
plt.ylabel("Liczba recenzji")
plt.show()

# ==========================================
# KROK 3: Analiza długości tekstów i odsiew
# ==========================================
print("\nObliczanie długości recenzji (w słowach)...")
df['word_count'] = df['full_review'].apply(lambda x: len(str(x).split()))

# Rysowanie histogramu długości
print("Rysowanie histogramu długości (ZAMKNIJ OKNO WYKRESU, ABY KOD SZEDŁ DALEJ)...")
plt.figure(figsize=(10, 5))
sns.histplot(df[df['word_count'] < 200]['word_count'], bins=50, kde=True, color='purple')
plt.title("Rozkład długości recenzji (do 200 słów)")
plt.xlabel("Liczba słów")
plt.ylabel("Częstotliwość")
plt.show()

# Odrzucamy recenzje poniżej 5 słów
print(f"Liczba wierszy przed usunięciem krótkich: {len(df)}")
df = df[df['word_count'] >= 5]
print(f"Liczba wierszy po usunięciu recenzji < 5 słów: {len(df)}")

# ==========================================
# KROK 4: Balansowanie klas (Undersampling)
# ==========================================
print("\n--- Balansowanie klas ---")
najmniejsza_klasa_ile = df['rating'].value_counts().min()
print(f"Najmniej liczna klasa ma {najmniejsza_klasa_ile} wierszy. Równamy do niej.")

# Balansowanie
df_balanced = df.groupby('rating').sample(n=najmniejsza_klasa_ile, random_state=42)

# Rysowanie wykresu po zbalansowaniu
print("Rysowanie zbalansowanego wykresu ocen (ZAMKNIJ OKNO WYKRESU, ABY KOD SZEDŁ DALEJ)...")
plt.figure(figsize=(8, 5))
sns.countplot(data=df_balanced, x='rating', palette='viridis')
plt.title("Rozkład ocen PO zbalansowaniu (Idealnie równy)")
plt.xlabel("Ocena (Gwiazdki)")
plt.ylabel("Liczba recenzji")
plt.show()

# ==========================================
# KROK 5: Podgląd z uwzględnieniem TYTUŁÓW
# ==========================================
print("\n--- PRZEGLĄD PRÓBKI (po 3 recenzje dla każdej oceny) ---")
podglad = df_balanced.groupby('rating').sample(n=3, random_state=7)

for ocena in sorted(podglad['rating'].unique()):
    print(f"\n{'=' * 25} OCENA: {ocena} GWIAZDKI {'=' * 25}")
    sub_df = podglad[podglad['rating'] == ocena]

    for i, (_, row) in enumerate(sub_df.iterrows(), 1):
        tytul = str(row['title'])
        tekst = str(row['full_review'])
        skrocony_tekst = (tekst[:250] + ' [...więcej]') if len(tekst) > 250 else tekst

        print(f"{i}. [TYTUŁ]: {tytul}")
        print(f"   [TREŚĆ]: {skrocony_tekst}\n")

# ==========================================
# KROK 6: ZAPIS GOTOWYCH DANYCH
# ==========================================
if 'word_count' in df_balanced.columns:
    df_balanced = df_balanced.drop(columns=['word_count'])

df_balanced.to_parquet(PLIK_WYJSCIOWY)
print(f"Zapisano czysty i zbalansowany zbiór do pliku na przyszłość: {PLIK_WYJSCIOWY}")

# ==========================================
# KROK 7: ZAMIANA TEKSTU NA LICZBY (Wektoryzacja TF-IDF)
# ==========================================
print("\n--- Przechodzę do analizy NLP ---")
print("Zamieniam słowa na macierz matematyczną (może chwilę zająć)...")

# Bierzemy tylko 2000 najpopularniejszych słów z całego zbioru
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')

# Wykorzystujemy zbalansowane dane (df_balanced), które mamy już w pamięci
X = vectorizer.fit_transform(df_balanced['full_review'])
y = df_balanced['rating']

# ==========================================
# KROK 8: SZUKANIE KORELACJI (Regresja Liniowa)
# ==========================================
print("Trenuję prosty model liniowy do znalezienia korelacji...")
model = LinearRegression()
model.fit(X, y)

# ==========================================
# KROK 9: ANALIZA WYNIKÓW I RYSOWANIE
# ==========================================
slowa = vectorizer.get_feature_names_out()
wagi = model.coef_

df_korelacja = pd.DataFrame({
    'slowo': slowa,
    'wplyw_na_ocene': wagi
})

top_pozytywne = df_korelacja.sort_values(by='wplyw_na_ocene', ascending=False).head(15)
top_negatywne = df_korelacja.sort_values(by='wplyw_na_ocene', ascending=True).head(15)

print("\n--- TOP 5 SŁÓW POZYTYWNYCH ---")
print(top_pozytywne.head())

print("\n--- TOP 5 SŁÓW NEGATYWNYCH ---")
print(top_negatywne.head())

print("\nRysowanie wykresów wpływu słów na ocenę...")
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

sns.barplot(data=top_pozytywne, x='wplyw_na_ocene', y='slowo', ax=axes[0], palette='Greens_r')
axes[0].set_title("Słowa podnoszące ocenę (Korelacja pozytywna)", fontsize=14)
axes[0].set_xlabel("Waga (Siła wpływu)")
axes[0].set_ylabel("Słowo")

sns.barplot(data=top_negatywne, x='wplyw_na_ocene', y='slowo', ax=axes[1], palette='Reds_r')
axes[1].set_title("Słowa obniżające ocenę (Korelacja negatywna)", fontsize=14)
axes[1].set_xlabel("Waga (Siła wpływu)")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()

print("\n--- SKRYPT ZAKOŃCZONY POMYŚLNIE ---")