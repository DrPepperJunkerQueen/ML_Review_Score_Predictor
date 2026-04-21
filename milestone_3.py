import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# Ustawienie stylu wykresów
sns.set_theme(style="whitegrid")

# ==========================================
# 1. ŁADOWANIE DANYCH
# ==========================================
print("Ładowanie zbalansowanych danych...")
df = pd.read_parquet("zbalansowane_gotowe_do_treningu.parquet")

# ==========================================
# 2. ZAMIANA TEKSTU NA LICZBY (Wersja ulepszona)
# ==========================================
print("Zamieniam słowa na macierz matematyczną (uwzględniam pary słów i przeczenia)...")

# Wyciągamy domyślną listę angielskich śmieciowych słów...
lista_stop_words = set(ENGLISH_STOP_WORDS)
# ...i WYRZUCAMY z niej słowa, które zmieniają sens na negatywny!
# Dzięki temu model ich nie usunie z recenzji.
wazne_przeczenia = {"no", "not", "nor", "none", "nothing", "without", "doesn", "isn", "wasn", "couldn", "wouldn"}
bezpieczne_stop_words = list(lista_stop_words - wazne_przeczenia)

# ngram_range=(1, 3) oznacza, że bierzemy pojedyncze słowa, pary (bigramy) i trójki (trigramy)
# max_features podnosimy lekko do 3000, bo doszło nam sporo nowych kombinacji słów
vectorizer = TfidfVectorizer(
    max_features=3000, 
    stop_words=bezpieczne_stop_words, 
    ngram_range=(1, 3) 
)

X = vectorizer.fit_transform(df['full_review'])
y = df['rating']

# ==========================================
# 3. SZUKANIE KORELACJI (Regresja Liniowa)
# ==========================================
print("Trenuję prosty model liniowy do znalezienia korelacji...")
model = LinearRegression()
model.fit(X, y)

# ==========================================
# 4. ANALIZA WYNIKÓW
# ==========================================
# Pobieramy listę 2000 słów i wagę (korelację), jaką model przypisał każdemu z nich
slowa = vectorizer.get_feature_names_out()
wagi = model.coef_

# Tworzymy wygodną tabelę
df_korelacja = pd.DataFrame({
    'slowo': slowa, 
    'wplyw_na_ocene': wagi
})

# Wyciągamy 15 słów, które najbardziej podbijają ocenę (w stronę 5 gwiazdek)
top_pozytywne = df_korelacja.sort_values(by='wplyw_na_ocene', ascending=False).head(15)

# Wyciągamy 15 słów, które najbardziej zaniżają ocenę (w stronę 1 gwiazdki)
top_negatywne = df_korelacja.sort_values(by='wplyw_na_ocene', ascending=True).head(15)

print("\n--- TOP 5 SŁÓW POZYTYWNYCH ---")
print(top_pozytywne.head())

print("\n--- TOP 5 SŁÓW NEGATYWNYCH ---")
print(top_negatywne.head())

# ==========================================
# 5. RYSOWANIE WYKRESÓW
# ==========================================
print("\nRysowanie wykresów...")
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Wykres pozytywnych
sns.barplot(data=top_pozytywne, x='wplyw_na_ocene', y='slowo', ax=axes[0], palette='Greens_r')
axes[0].set_title("Słowa podnoszące ocenę (Korelacja pozytywna)", fontsize=14)
axes[0].set_xlabel("Waga (Siła wpływu)")
axes[0].set_ylabel("Słowo")

# Wykres negatywnych
sns.barplot(data=top_negatywne, x='wplyw_na_ocene', y='slowo', ax=axes[1], palette='Reds_r')
axes[1].set_title("Słowa obniżające ocenę (Korelacja negatywna)", fontsize=14)
axes[1].set_xlabel("Waga (Siła wpływu)")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()