import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ustawienie ładniejszego stylu wykresów
sns.set_theme(style="whitegrid")

PLIK_WEJSCIOWY = "przetworzona_probka_100k.parquet"
PLIK_WYJSCIOWY = "zbalansowane_gotowe_do_treningu.parquet"

# ==========================================
# SPRAWDZENIE ZALEŻNOŚCI (Czy Kamień 1 jest zrobiony?)
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
print("\nRysowanie wykresu początkowego rozkładu ocen...")
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
    print(f"\n{'='*25} OCENA: {ocena} GWIAZDKI {'='*25}")
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

print(f"--- GOTOWE ---")
print(f"Zapisano czysty i zbalansowany zbiór do pliku: {PLIK_WYJSCIOWY}")