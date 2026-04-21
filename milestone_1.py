import os
import pandas as pd
from datasets import load_dataset

# Wskazujemy dysk z dużą ilością miejsca (zmień na swój, jeśli trzeba)
CACHE_DIR = "C:/huggingface_cache" 
# Nazwa pliku, do którego zapiszemy gotowe dane
OUTPUT_FILE = "przetworzona_probka_100k.parquet"

# Sprawdzamy, czy wykonaliśmy już wcześniej tę pracę
if os.path.exists(OUTPUT_FILE):
    print(f"Znaleziono gotowy plik '{OUTPUT_FILE}'! Błyskawiczne ładowanie z dysku...")
    # Wczytujemy dane od razu do tabeli
    df_sample = pd.read_parquet(OUTPUT_FILE)

else:
    print("Brak przetworzonych danych na dysku. Uruchamiam pełen proces...")
    
    # 1. Pobieranie gier
    print("Ładowanie recenzji gier...")
    games = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023", 
        "raw_review_Video_Games", 
        split="full",
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
    # Wyciągamy od razu ocenę, tytuł i treść
    df_games = games.to_pandas()[['rating', 'title', 'text']]

    # 2. Pobieranie filmów i seriali
    print("Ładowanie recenzji filmów...")
    movies = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023", 
        "raw_review_Movies_and_TV", 
        split="full",
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    )
    df_movies = movies.to_pandas()[['rating', 'title', 'text']]

    # 3. Łączenie w jedną dużą bazę
    print("Łączenie baz danych...")
    df_all = pd.concat([df_games, df_movies], ignore_index=True)

    # 4. Czyszczenie danych (Data Preprocessing)
    print("Czyszczenie pustych wierszy...")
    # Usuwamy wiersze, w których brakuje oceny, tytułu lub tekstu (żeby nie psuły nauki modelu)
    df_all = df_all.dropna(subset=['rating', 'title', 'text'])

    # Łączymy tytuł i treść recenzji kropką, tworząc pełny tekst dla sztucznej inteligencji
    df_all['full_review'] = df_all['title'] + ". " + df_all['text']

    # 5. Próbkowanie (losowanie mniejszej ilości danych do eksperymentów)
    print("Losowanie próbki treningowej...")
    df_sample = df_all.sample(n=100000, random_state=42)
    
    # ZAPIS GOTOWYCH DANYCH NA PRZYSZŁOŚĆ
    print(f"Zapisywanie czystej próbki do pliku '{OUTPUT_FILE}'...")
    df_sample.to_parquet(OUTPUT_FILE)


# 6. Podgląd ostatecznego wyniku
print("\n--- GOTOWE DANE ---")
print(f"Łączna liczba recenzji w próbce: {len(df_sample)}")
# Wyświetlamy samą ocenę i połączony tekst z 5 pierwszych wierszy
print(df_sample[['rating', 'full_review']].head())