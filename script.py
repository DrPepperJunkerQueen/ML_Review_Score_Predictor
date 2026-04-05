import pandas as pd
from datasets import load_dataset

# Wskazujemy dysk z dużą ilością miejsca (zmień na swój, jeśli trzeba)
CACHE_DIR = "D:/huggingface_cache" 

# 1. Pobieranie gier
print("Pobieranie recenzji gier...")
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
print("Pobieranie recenzji filmów...")
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
# Na początek bierzemy losowe 100 000 wierszy, żeby testy trwały minuty, a nie dnie.
print("Losowanie próbki treningowej...")
df_sample = df_all.sample(n=100000, random_state=42)


# 6. Podgląd ostatecznego wyniku
print("\n--- GOTOWE DANE ---")
print(f"Łączna liczba recenzji w próbce: {len(df_sample)}")
# Wyświetlamy samą ocenę i połączony tekst z 5 pierwszych wierszy
print(df_sample[['rating', 'full_review']].head())