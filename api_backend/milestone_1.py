import os
import pandas as pd
from datasets import load_dataset, concatenate_datasets

CACHE_DIR = input("Wprowadz poprawna sciezke do bazy danych (np. C:/) : ")   #S:/huggingface_cache
OUTPUT_FILE = "przetworzona_probka_500k.parquet"

if os.path.exists(OUTPUT_FILE):
    print(f"Znaleziono gotowy plik '{OUTPUT_FILE}'! Błyskawiczne ładowanie z dysku...")
    df_sample = pd.read_parquet(OUTPUT_FILE)

else:
    print("Brak przetworzonych danych na dysku. Uruchamiam pełen proces w trybie oszczędzania RAM...")

    # 1. Pobieranie gier (tylko wskazane kolumny, bez ładowania całosci do RAM)
    print("Ładowanie recenzji gier...")
    games = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_Video_Games",
        split="full",
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    ).select_columns(['rating', 'title', 'text'])

    # 2. Pobieranie filmów i seriali
    print("Ładowanie recenzji filmów...")
    movies = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_Movies_and_TV",
        split="full",
        trust_remote_code=True,
        cache_dir=CACHE_DIR
    ).select_columns(['rating', 'title', 'text'])

    # 3. Łączenie baz danych na poziomie dysku (Hugging Face)
    print("Łączenie baz danych...")
    all_data = concatenate_datasets([games, movies])

    # 4. PRÓBKOWANIE PRZED ZAŁADOWANIEM DO RAM
    print("Losowanie próbki (550k żeby mieć zapas na puste wiersze)...")
    # Losujemy 550 000 wierszy (szybkie i tanie dla pamięci)
    sampled_data = all_data.shuffle(seed=42).select(range(550000))

    # 5. Konwersja tylko MAŁEJ próbki do Pandas
    print("Konwersja próbki do tabeli Pandas...")
    df_all = sampled_data.to_pandas()

    # 6. Czyszczenie danych
    print("Czyszczenie pustych wierszy i formatowanie...")
    df_all = df_all.dropna(subset=['rating', 'title', 'text'])
    df_all['full_review'] = df_all['title'] + ". " + df_all['text']

    # Ucinamy dokładnie do 500 000 (ponieważ wcześniej wzięliśmy lekki zapas na wypadek usuniętych NaN)
    df_sample = df_all.head(500000)

    # 7. ZAPIS GOTOWYCH DANYCH NA PRZYSZŁOŚĆ
    print(f"Zapisywanie czystej próbki do pliku '{OUTPUT_FILE}'...")
    df_sample.to_parquet(OUTPUT_FILE)

# 8. Podgląd ostatecznego wyniku
print("\n--- GOTOWE DANE ---")
print(f"Łączna liczba recenzji w próbce: {len(df_sample)}")
print(df_sample[['rating', 'full_review']].head())