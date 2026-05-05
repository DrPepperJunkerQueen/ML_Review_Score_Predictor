import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix

sns.set_theme(style="whitegrid")

# ══════════════════════════════════════════════════════════════
# POMOCNICZE FUNKCJE
# ══════════════════════════════════════════════════════════════

SKALA = 5.0 - 1.0  # rozpiętość = 4.0

def raportuj(krok: int, total: int, opis: str, czas_start: float, wynik: str = ""):
    """Drukuje czytelny wiersz postępu z procentem i czasem elapsed."""
    elapsed  = time.time() - czas_start
    procent  = krok / total * 100
    pasek_len = 20
    wypelnione = int(pasek_len * krok / total)
    pasek    = "█" * wypelnione + "░" * (pasek_len - wypelnione)
    czas_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
    suffix   = f"  →  {wynik}" if wynik else ""
    print(f"  [{krok:>{len(str(total))}}/{total}] {pasek} {procent:5.1f}% | {czas_str} elapsed | {opis}{suffix}")

def proximity_accuracy(y_true, y_pred):
    """Skuteczność uwzględniająca bliskość: 1.0 = ideał, 0.0 = max pomyłka."""
    return float(np.mean(1.0 - np.abs(np.clip(y_pred, 1.0, 5.0) - y_true) / SKALA))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.clip(y_pred, 1.0, 5.0) - y_true)))

def do_przedzialow(wartosci):
    """Zamienia oceny float na przedziały: 1→'1.0–1.9', 2→'2.0–2.9' itd."""
    bins   = [1.0, 2.0, 3.0, 4.0, 5.0, 5.01]
    labels = ['1.0–1.9', '2.0–2.9', '3.0–3.9', '4.0–4.9', '5.0']
    return pd.cut(np.clip(wartosci, 1.0, 5.0), bins=bins, labels=labels, right=False)

# ══════════════════════════════════════════════════════════════
# KROK 1: ŁADOWANIE DANYCH
# ══════════════════════════════════════════════════════════════
print("Ładowanie danych...")
if not os.path.exists("zbalansowane_gotowe_do_treningu.parquet"):
    print("BŁĄD: Nie znaleziono pliku 'zbalansowane_gotowe_do_treningu.parquet'!")
    print("Uruchom najpierw Milestone 2.")
    exit()

df = pd.read_parquet("zbalansowane_gotowe_do_treningu.parquet")
X = df['full_review']
y = df['rating'].astype(float)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ══════════════════════════════════════════════════════════════
# KROK 2: WEKTORYZACJA
# ══════════════════════════════════════════════════════════════
print("Wektoryzacja TF-IDF (5000 cech)...")
t0 = time.time()
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = vectorizer.fit_transform(X_train_raw)
X_test  = vectorizer.transform(X_test_raw)
print(f"  Gotowe ({time.time()-t0:.1f}s)  |  Zbiór treningowy: {X_train.shape[0]:,} wierszy, {X_train.shape[1]:,} cech")

# ══════════════════════════════════════════════════════════════
# KROK 3: PORÓWNANIE MODELI
# ══════════════════════════════════════════════════════════════
PROBKA = 30_000

print("\n" + "="*60)
print("KROK 3: PORÓWNANIE MODELI")
print("="*60)

kandydaci = [
    ("Ridge (α=1.0)",            Ridge(alpha=1.0),                                        False),
    ("SVR (C=1.0)",              SVR(C=1.0, kernel='rbf'),                                True),
    ("RandomForest (100 drzew)", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), False),
]

wyniki_porownania = []
total_k3 = len(kandydaci)
t_k3 = time.time()

for i, (nazwa, model, uzyj_probki) in enumerate(kandydaci, 1):
    if uzyj_probki:
        idx = np.random.default_rng(42).choice(X_train.shape[0], PROBKA, replace=False)
        Xt, yt = X_train[idx], y_train.iloc[idx]
        info_probka = f"próbka {PROBKA:,}"
    else:
        Xt, yt = X_train, y_train
        info_probka = f"pełny zbiór {X_train.shape[0]:,}"

    raportuj(i - 1, total_k3, f"Trenuję: {nazwa} ({info_probka})", t_k3)
    t_model = time.time()
    model.fit(Xt, yt)
    pred = model.predict(X_test)
    acc  = proximity_accuracy(y_test.values, pred)
    err  = mae(y_test.values, pred)
    wyniki_porownania.append({'Model': nazwa, 'Proximity Accuracy': acc, 'MAE (gwiazdki)': err})
    raportuj(i, total_k3, f"{nazwa}", t_k3,
             wynik=f"Accuracy={acc*100:.2f}%  MAE={err:.4f}  (czas modelu: {time.time()-t_model:.1f}s)")

df_por = pd.DataFrame(wyniki_porownania).sort_values('Proximity Accuracy', ascending=False)
print(f"\n--- TABELA PORÓWNAWCZA (łączny czas: {time.time()-t_k3:.1f}s) ---")
print(df_por.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Porównanie modeli", fontsize=15, fontweight='bold')
sns.barplot(data=df_por, x='Model', y='Proximity Accuracy', ax=axes[0], palette='Blues_d')
axes[0].set_title("Proximity Accuracy (wyżej = lepiej)")
axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
axes[0].set_ylim(0, 1)
axes[0].set_xlabel("")
sns.barplot(data=df_por, x='Model', y='MAE (gwiazdki)', ax=axes[1], palette='Oranges_d')
axes[1].set_title("MAE — Średni błąd bezwzględny (niżej = lepiej)")
axes[1].set_xlabel("")
plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════
# KROK 4: TUNING NAJLEPSZEGO MODELU
# ══════════════════════════════════════════════════════════════
najlepszy_nazwa = df_por.iloc[0]['Model']
print("\n" + "="*60)
print(f"KROK 4: TUNING → {najlepszy_nazwa}")
print("="*60)

wyniki_tuning = []

if najlepszy_nazwa.startswith("Ridge"):
    parametry = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    etykiety  = [f"α={a}" for a in parametry]
    param_label = "alpha (Ridge)"
    def buduj_model(p): return Ridge(alpha=p)
    uzyj_probki_tuning = False

elif najlepszy_nazwa.startswith("SVR"):
    parametry = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    etykiety  = [f"C={c}" for c in parametry]
    param_label = "C (SVR)"
    def buduj_model(p): return SVR(C=p, kernel='rbf')
    uzyj_probki_tuning = True

else:
    parametry = [50, 100, 200, 300, 500]
    etykiety  = [f"n={n}" for n in parametry]
    param_label = "n_estimators (RandomForest)"
    def buduj_model(p): return RandomForestRegressor(n_estimators=p, random_state=42, n_jobs=-1)
    uzyj_probki_tuning = False

if uzyj_probki_tuning:
    idx_t = np.random.default_rng(42).choice(X_train.shape[0], PROBKA, replace=False)
    Xt_t, yt_t = X_train[idx_t], y_train.iloc[idx_t]
    print(f"  (SVR — używam próbki {PROBKA:,} wierszy dla szybkości)")
else:
    Xt_t, yt_t = X_train, y_train

total_k4 = len(parametry)
t_k4 = time.time()

for i, (p, etykieta) in enumerate(zip(parametry, etykiety), 1):
    raportuj(i - 1, total_k4, f"Testuję {etykieta}", t_k4)
    t_p = time.time()
    m    = buduj_model(p)
    m.fit(Xt_t, yt_t)
    pred = m.predict(X_test)
    acc  = proximity_accuracy(y_test.values, pred)
    err  = mae(y_test.values, pred)
    wyniki_tuning.append({'Parametr': etykieta, 'Proximity Accuracy': acc, 'MAE': err})
    raportuj(i, total_k4, f"{etykieta}", t_k4,
             wynik=f"Accuracy={acc*100:.2f}%  MAE={err:.4f}  ({time.time()-t_p:.1f}s)")

df_tuning = pd.DataFrame(wyniki_tuning)
najlepszy_param = df_tuning.loc[df_tuning['Proximity Accuracy'].idxmax(), 'Parametr']
print(f"\n★ Najlepszy parametr: {najlepszy_param}  (łączny czas tuningu: {time.time()-t_k4:.1f}s)")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_tuning['Parametr'], df_tuning['Proximity Accuracy'], marker='o', color='steelblue', label='Proximity Accuracy')
ax2 = ax.twinx()
ax2.plot(df_tuning['Parametr'], df_tuning['MAE'], marker='s', color='coral', linestyle='--', label='MAE')
ax.set_title(f"Tuning parametru: {param_label}", fontsize=13)
ax.set_xlabel("Wartość parametru")
ax.set_ylabel("Proximity Accuracy", color='steelblue')
ax2.set_ylabel("MAE (gwiazdki)", color='coral')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════
# KROK 5: FINALNY MODEL
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("KROK 5: FINALNY MODEL Z NAJLEPSZYM PARAMETREM")
print("="*60)

# Wyciągamy wartość liczbową z etykiety (np. "α=5.0" → 5.0, "n=200" → 200)
val_str = najlepszy_param.split('=')[1]

print(f"  Trenuję finalny model: {najlepszy_nazwa} [{najlepszy_param}]...")
t_fin = time.time()

if najlepszy_nazwa.startswith("Ridge"):
    finalny_model = Ridge(alpha=float(val_str)).fit(X_train, y_train)
elif najlepszy_nazwa.startswith("SVR"):
    finalny_model = SVR(C=float(val_str), kernel='rbf').fit(X_train[idx_t], y_train.iloc[idx_t])
else:
    finalny_model = RandomForestRegressor(n_estimators=int(val_str), random_state=42, n_jobs=-1).fit(X_train, y_train)

print(f"  Trening zakończony ({time.time()-t_fin:.1f}s)")

y_pred_final = np.clip(finalny_model.predict(X_test), 1.0, 5.0)
acc_final  = proximity_accuracy(y_test.values, y_pred_final)
mae_final  = mae(y_test.values, y_pred_final)
rmse_final = float(np.sqrt(np.mean((y_pred_final - y_test.values)**2)))

print(f"\n  Finalny model:          {najlepszy_nazwa} [{najlepszy_param}]")
print(f"  Proximity Accuracy:     {acc_final*100:.2f}%")
print(f"  MAE:                    {mae_final:.4f} gwiazdki")
print(f"  RMSE:                   {rmse_final:.4f} gwiazdki")

# ══════════════════════════════════════════════════════════════
# KROK 6: HEATMAPA PRZEDZIAŁÓW
# ══════════════════════════════════════════════════════════════
print("\nRysowanie heatmapy przedziałów (zamknij okno, aby kontynuować)...")

przedzial_labels = ['1.0–1.9', '2.0–2.9', '3.0–3.9', '4.0–4.9', '5.0']
rzeczywiste  = do_przedzialow(y_test.values)
przewidywane = do_przedzialow(y_pred_final)

cm = confusion_matrix(rzeczywiste, przewidywane, labels=przedzial_labels)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Macierz pomyłek — przedziały ocen", fontsize=14, fontweight='bold')
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=przedzial_labels, yticklabels=przedzial_labels, ax=axes[0])
axes[0].set_title("Liczba próbek")
axes[0].set_xlabel("Przedział przewidywany")
axes[0].set_ylabel("Przedział rzeczywisty")
sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=przedzial_labels, yticklabels=przedzial_labels, ax=axes[1])
axes[1].set_title("Rozkład procentowy (wiersz = 100%)")
axes[1].set_xlabel("Przedział przewidywany")
axes[1].set_ylabel("")
plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════
# KROK 7: ANALIZA BŁĘDÓW PER KATEGORIA
# ══════════════════════════════════════════════════════════════
print("Rysowanie analizy błędów per kategoria (zamknij okno, aby kontynuować)...")

df_analiza = pd.DataFrame({
    'rzeczywista':  y_test.values,
    'przewidywana': y_pred_final,
    'blad_abs':     np.abs(y_pred_final - y_test.values),
    'skutecznosc':  1.0 - np.abs(y_pred_final - y_test.values) / SKALA,
})
df_analiza['ocena_int'] = df_analiza['rzeczywista'].round().astype(int)
stats = df_analiza.groupby('ocena_int').agg(
    sredni_blad=('blad_abs', 'mean'),
    skutecznosc_avg=('skutecznosc', 'mean'),
    liczba=('blad_abs', 'count'),
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Analiza błędów per kategoria oceny", fontsize=13, fontweight='bold')
sns.barplot(data=stats, x='ocena_int', y='sredni_blad', ax=axes[0], palette='Reds_d')
axes[0].set_title("Średni błąd bezwzględny")
axes[0].set_xlabel("Rzeczywista ocena (gwiazdki)")
axes[0].set_ylabel("MAE (gwiazdki)")
sns.barplot(data=stats, x='ocena_int', y='skutecznosc_avg', ax=axes[1], palette='Greens_d')
axes[1].set_title("Średnia skuteczność")
axes[1].set_xlabel("Rzeczywista ocena (gwiazdki)")
axes[1].set_ylabel("Proximity Accuracy")
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
axes[1].set_ylim(0, 1)
plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════
# KROK 8: KRZYWA UCZENIA SIĘ (ręczna pętla z postępem)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("KROK 8: KRZYWA UCZENIA SIĘ")
print("="*60)

# SVR i RandomForest są zbyt wolne dla 10 punktów × 3 foldy — używamy Ridge
if najlepszy_nazwa.startswith("Ridge"):
    model_krzywej = Ridge(alpha=float(val_str))
    print(f"  Model: Ridge [α={val_str}]")
else:
    model_krzywej = Ridge(alpha=1.0)
    print(f"  (Oryginalny model zbyt wolny dla krzywej uczenia — używam Ridge α=1.0 jako proxy)")

N_PUNKTOW = 10
N_FOLDOW  = 3
udzialy   = np.linspace(0.05, 1.0, N_PUNKTOW)
kf        = KFold(n_splits=N_FOLDOW, shuffle=True, random_state=42)

train_sizes_abs = []
train_maes_wszystkie = []
test_maes_wszystkie  = []

total_k8 = N_PUNKTOW * N_FOLDOW
krok_k8  = 0
t_k8     = time.time()

for ui, udzial in enumerate(udzialy):
    n_prob = int(X_train.shape[0] * udzial)
    idx_u  = np.random.default_rng(ui).choice(X_train.shape[0], n_prob, replace=False)
    X_u    = X_train[idx_u]
    y_u    = y_train.iloc[idx_u].values

    fold_train_mae = []
    fold_test_mae  = []

    for fi, (tr_idx, val_idx) in enumerate(kf.split(X_u), 1):
        krok_k8 += 1
        raportuj(krok_k8 - 1, total_k8,
                 f"Krzywa: {udzial*100:.0f}% danych ({n_prob:,} wierszy), fold {fi}/{N_FOLDOW}",
                 t_k8)

        t_fold = time.time()
        m = Ridge(alpha=model_krzywej.alpha if hasattr(model_krzywej, 'alpha') else 1.0)
        m.fit(X_u[tr_idx], y_u[tr_idx])

        p_train = m.predict(X_u[tr_idx])
        p_val   = m.predict(X_u[val_idx])
        fold_train_mae.append(mae(y_u[tr_idx], p_train))
        fold_test_mae.append(mae(y_u[val_idx], p_val))

        raportuj(krok_k8, total_k8,
                 f"Krzywa: {udzial*100:.0f}% danych ({n_prob:,} wierszy), fold {fi}/{N_FOLDOW}",
                 t_k8,
                 wynik=f"train_MAE={fold_train_mae[-1]:.4f}  val_MAE={fold_test_mae[-1]:.4f}  ({time.time()-t_fold:.1f}s)")

    train_sizes_abs.append(n_prob)
    train_maes_wszystkie.append(fold_train_mae)
    test_maes_wszystkie.append(fold_test_mae)

print(f"\n  Krzywa uczenia zakończona (łączny czas: {time.time()-t_k8:.1f}s)")

train_mae_avg = np.array([np.mean(v) for v in train_maes_wszystkie])
train_mae_std = np.array([np.std(v)  for v in train_maes_wszystkie])
test_mae_avg  = np.array([np.mean(v) for v in test_maes_wszystkie])
test_mae_std  = np.array([np.std(v)  for v in test_maes_wszystkie])

train_acc_avg = 1 - train_mae_avg / SKALA
test_acc_avg  = 1 - test_mae_avg  / SKALA

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Krzywa uczenia się", fontsize=13, fontweight='bold')

axes[0].plot(train_sizes_abs, train_mae_avg, 'o-', color='steelblue', label='Zbiór treningowy')
axes[0].plot(train_sizes_abs, test_mae_avg,  's-', color='coral',     label='Zbiór walidacyjny')
axes[0].fill_between(train_sizes_abs,
                     train_mae_avg - train_mae_std, train_mae_avg + train_mae_std,
                     alpha=0.15, color='steelblue')
axes[0].fill_between(train_sizes_abs,
                     test_mae_avg - test_mae_std, test_mae_avg + test_mae_std,
                     alpha=0.15, color='coral')
axes[0].set_title("MAE vs liczba próbek treningowych")
axes[0].set_xlabel("Liczba próbek treningowych")
axes[0].set_ylabel("MAE (gwiazdki)")
axes[0].legend()

axes[1].plot(train_sizes_abs, train_acc_avg, 'o-', color='steelblue', label='Zbiór treningowy')
axes[1].plot(train_sizes_abs, test_acc_avg,  's-', color='coral',     label='Zbiór walidacyjny')
axes[1].fill_between(train_sizes_abs,
                     (1 - (train_mae_avg + train_mae_std) / SKALA),
                     (1 - (train_mae_avg - train_mae_std) / SKALA),
                     alpha=0.15, color='steelblue')
axes[1].fill_between(train_sizes_abs,
                     (1 - (test_mae_avg + test_mae_std) / SKALA),
                     (1 - (test_mae_avg - test_mae_std) / SKALA),
                     alpha=0.15, color='coral')
axes[1].set_title("Proximity Accuracy vs liczba próbek")
axes[1].set_xlabel("Liczba próbek treningowych")
axes[1].set_ylabel("Proximity Accuracy")
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
axes[1].legend()

plt.tight_layout()
plt.show()

print("\n--- MILESTONE 4 ZAKOŃCZONY POMYŚLNIE ---")