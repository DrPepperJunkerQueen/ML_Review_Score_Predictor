import os
import sys
import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib  # <-- DODANO IMPORT
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix

sns.set_theme(style="whitegrid")

# ══════════════════════════════════════════════════════════════
# POMOCNICZE FUNKCJE
# ══════════════════════════════════════════════════════════════

SKALA = 5.0 - 1.0  # rozpiętość = 4.0

def pasek_postepu(krok: int, total: int, opis: str, czas_start: float,
                  wynik: str = "", dlugosc: int = 30):
    """Rysuje jeden wiersz postępu, nadpisując poprzedni (\\r)."""
    elapsed    = time.time() - czas_start
    procent    = krok / total * 100
    wypelnione = int(dlugosc * krok / total)
    pasek      = "█" * wypelnione + "░" * (dlugosc - wypelnione)
    czas_str   = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
    suffix     = f"  →  {wynik}" if wynik else ""
    # \r wraca na początek linii, \033[K czyści resztę
    print(f"\r  {pasek} {procent:5.1f}% | {czas_str} | {opis}{suffix}\033[K", end="", flush=True)

def nowa_linia():
    """Przechodzi do nowej linii po zakończeniu paska."""
    print()

def spinner_svr(czas_start: float, stop_event: threading.Event):
    """Kręci spinnerem dopóki SVR się trenuje (nie ma iteracji z zewnątrz)."""
    klatki = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    i = 0
    while not stop_event.is_set():
        elapsed  = time.time() - czas_start
        czas_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        print(f"\r  {klatki[i % len(klatki)]} SVR trenuje... | {czas_str} elapsed (brak iteracji — czarna skrzynka)\033[K",
              end="", flush=True)
        i += 1
        time.sleep(0.1)

def proximity_accuracy(y_true, y_pred):
    return float(np.mean(1.0 - np.abs(np.clip(y_pred, 1.0, 5.0) - y_true) / SKALA))

def mae_score(y_true, y_pred):
    return float(np.mean(np.abs(np.clip(y_pred, 1.0, 5.0) - y_true)))

def do_przedzialow(wartosci):
    bins   = [1.0, 2.0, 3.0, 4.0, 5.0, 5.01]
    labels = ['1.0–1.9', '2.0–2.9', '3.0–3.9', '4.0–4.9', '5.0']
    return pd.cut(np.clip(wartosci, 1.0, 5.0), bins=bins, labels=labels, right=False)

# ══════════════════════════════════════════════════════════════
# TRENING Z POSTĘPEM — osobna funkcja dla każdego typu modelu
# ══════════════════════════════════════════════════════════════

def trenuj_sgd(X_tr, y_tr, n_epok=100, alpha=0.0001):
    model = SGDRegressor(
        loss='squared_error', penalty='l2', alpha=alpha,
        max_iter=1, warm_start=True, random_state=42, tol=None
    )
    t = time.time()
    for epoka in range(1, n_epok + 1):
        model.partial_fit(X_tr, y_tr)
        pasek_postepu(epoka, n_epok,
                      f"SGD/Ridge  epoka {epoka:>{len(str(n_epok))}}/{n_epok}", t)
    nowa_linia()
    return model

# KLASA WYCIĄGNIĘTA DO ZASIĘGU GLOBALNEGO (ABY DZIAŁAŁ ZAPIS)
class RFWrapper:
    def __init__(self, drzewa): 
        self.drzewa = drzewa
    def predict(self, X):
        return np.mean([d.predict(X) for d in self.drzewa], axis=0)

def trenuj_rf(X_tr, y_tr, n_drzew=100):
    drzewa = []
    t = time.time()
    for i in range(1, n_drzew + 1):
        drzewo = DecisionTreeRegressor(
            max_depth=None,
            max_features='sqrt',
            random_state=i
        )
        idx = np.random.default_rng(i).integers(0, X_tr.shape[0], X_tr.shape[0])
        drzewo.fit(X_tr[idx], y_tr.iloc[idx] if hasattr(y_tr, 'iloc') else y_tr[idx])
        drzewa.append(drzewo)
        pasek_postepu(i, n_drzew, f"RandomForest  drzewo {i:>{len(str(n_drzew))}}/{n_drzew}", t)
    nowa_linia()

    return RFWrapper(drzewa)

def trenuj_svr(X_tr, y_tr, C=1.0):
    stop = threading.Event()
    t    = time.time()
    w    = threading.Thread(target=spinner_svr, args=(t, stop), daemon=True)
    w.start()
    model = SVR(C=C, kernel='rbf')
    model.fit(X_tr, y_tr)
    stop.set()
    w.join()
    elapsed = time.time() - t
    print(f"\r  ✓ SVR gotowy ({elapsed:.1f}s)\033[K")
    return model

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
print(f"  Gotowe ({time.time()-t0:.1f}s)  |  {X_train.shape[0]:,} wierszy, {X_train.shape[1]:,} cech")

# ══════════════════════════════════════════════════════════════
# KROK 3: PORÓWNANIE MODELI
# ══════════════════════════════════════════════════════════════
PROBKA   = 30_000
N_EPOK   = 100
N_DRZEW  = 100

print("\n" + "="*60)
print("KROK 3: PORÓWNANIE MODELI")
print("="*60)

wyniki_porownania = []

# --- Model 1: SGD/Ridge ---
print("\n[Model 1/3] SGD Regressor (odpowiednik Ridge) — pełny zbiór")
t_m = time.time()
m_sgd   = trenuj_sgd(X_train, y_train.values, n_epok=N_EPOK)
pred    = np.clip(m_sgd.predict(X_test), 1.0, 5.0)
acc_sgd = proximity_accuracy(y_test.values, pred)
mae_sgd = mae_score(y_test.values, pred)
wyniki_porownania.append({'Model': f'SGD/Ridge ({N_EPOK} epok)', 'Proximity Accuracy': acc_sgd, 'MAE (gwiazdki)': mae_sgd})
print(f"  Wynik: Accuracy={acc_sgd*100:.2f}%  MAE={mae_sgd:.4f}  (łącznie {time.time()-t_m:.1f}s)")

# --- Model 2: SVR (próbka) ---
print(f"\n[Model 2/3] SVR (RBF kernel) — próbka {PROBKA:,} wierszy")
idx_svr = np.random.default_rng(42).choice(X_train.shape[0], PROBKA, replace=False)
t_m     = time.time()
m_svr   = trenuj_svr(X_train[idx_svr], y_train.iloc[idx_svr])
pred    = np.clip(m_svr.predict(X_test), 1.0, 5.0)
acc_svr = proximity_accuracy(y_test.values, pred)
mae_svr = mae_score(y_test.values, pred)
wyniki_porownania.append({'Model': 'SVR (C=1.0)', 'Proximity Accuracy': acc_svr, 'MAE (gwiazdki)': mae_svr})
print(f"  Wynik: Accuracy={acc_svr*100:.2f}%  MAE={mae_svr:.4f}  (łącznie {time.time()-t_m:.1f}s)")

# --- Model 3: RandomForest ---
print(f"\n[Model 3/3] Random Forest — {N_DRZEW} drzew, pełny zbiór")
t_m    = time.time()
m_rf   = trenuj_rf(X_train, y_train, n_drzew=N_DRZEW)
pred   = np.clip(m_rf.predict(X_test), 1.0, 5.0)
acc_rf = proximity_accuracy(y_test.values, pred)
mae_rf = mae_score(y_test.values, pred)
wyniki_porownania.append({'Model': f'RandomForest ({N_DRZEW} drzew)', 'Proximity Accuracy': acc_rf, 'MAE (gwiazdki)': mae_rf})
print(f"  Wynik: Accuracy={acc_rf*100:.2f}%  MAE={mae_rf:.4f}  (łącznie {time.time()-t_m:.1f}s)")

df_por = pd.DataFrame(wyniki_porownania).sort_values('Proximity Accuracy', ascending=False)
print("\n--- TABELA PORÓWNAWCZA ---")
print(df_por.to_string(index=False))

# ZAPIS TRZECH MODELI BAZOWYCH (DODANO)
print("\n--- ZAPIS 3 PORÓWNYWANYCH MODELI ---")
joblib.dump(m_sgd, "model_1_sgd.joblib")
joblib.dump(m_svr, "model_2_svr.joblib")
joblib.dump(m_rf, "model_3_rf.joblib")
print("Zapisano pomyślnie: model_1_sgd.joblib, model_2_svr.joblib, model_3_rf.joblib")

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

if najlepszy_nazwa.startswith("SGD"):
    alphas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    param_label = "alpha L2 (SGD/Ridge)"
    print(f"  Testuję {len(alphas)} wartości alpha ({N_EPOK} epok każda)...\n")
    total_t = len(alphas)
    for ti, a in enumerate(alphas, 1):
        print(f"  [{ti}/{total_t}] alpha={a}")
        t_p = time.time()
        m   = trenuj_sgd(X_train, y_train.values, n_epok=N_EPOK, alpha=a)
        pred = np.clip(m.predict(X_test), 1.0, 5.0)
        acc  = proximity_accuracy(y_test.values, pred)
        err  = mae_score(y_test.values, pred)
        wyniki_tuning.append({'Parametr': f'α={a}', 'Proximity Accuracy': acc, 'MAE': err})
        print(f"  Wynik: Accuracy={acc*100:.2f}%  MAE={err:.4f}  ({time.time()-t_p:.1f}s)\n")
    best_alpha_sgd = alphas[np.argmax([r['Proximity Accuracy'] for r in wyniki_tuning])]

elif najlepszy_nazwa.startswith("SVR"):
    cs = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    param_label = "C (SVR)"
    print(f"  Testuję {len(cs)} wartości C (próbka {PROBKA:,})...\n")
    for ti, c in enumerate(cs, 1):
        print(f"  [{ti}/{len(cs)}] C={c}")
        t_p = time.time()
        m   = trenuj_svr(X_train[idx_svr], y_train.iloc[idx_svr], C=c)
        pred = np.clip(m.predict(X_test), 1.0, 5.0)
        acc  = proximity_accuracy(y_test.values, pred)
        err  = mae_score(y_test.values, pred)
        wyniki_tuning.append({'Parametr': f'C={c}', 'Proximity Accuracy': acc, 'MAE': err})
        print(f"  Wynik: Accuracy={acc*100:.2f}%  MAE={err:.4f}  ({time.time()-t_p:.1f}s)\n")

else:  # RandomForest
    n_lista = [50, 100, 150, 200, 250, 300]
    param_label = "n_estimators (RandomForest)"
    print(f"  Testuję {len(n_lista)} wartości n_drzew...\n")
    for ti, n in enumerate(n_lista, 1):
        print(f"  [{ti}/{len(n_lista)}] n_drzew={n}")
        t_p = time.time()
        m   = trenuj_rf(X_train, y_train, n_drzew=n)
        pred = np.clip(m.predict(X_test), 1.0, 5.0)
        acc  = proximity_accuracy(y_test.values, pred)
        err  = mae_score(y_test.values, pred)
        wyniki_tuning.append({'Parametr': f'n={n}', 'Proximity Accuracy': acc, 'MAE': err})
        print(f"  Wynik: Accuracy={acc*100:.2f}%  MAE={err:.4f}  ({time.time()-t_p:.1f}s)\n")

df_tuning      = pd.DataFrame(wyniki_tuning)
najlepszy_param = df_tuning.loc[df_tuning['Proximity Accuracy'].idxmax(), 'Parametr']
print(f"\n★ Najlepszy parametr: {najlepszy_param}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_tuning['Parametr'], df_tuning['Proximity Accuracy'], marker='o', color='steelblue', label='Proximity Accuracy')
ax2 = ax.twinx()
ax2.plot(df_tuning['Parametr'], df_tuning['MAE'], marker='s', color='coral', linestyle='--', label='MAE')
ax.set_title(f"Tuning: {param_label}", fontsize=13)
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

val_str = najlepszy_param.split('=')[1]

if najlepszy_nazwa.startswith("SGD"):
    print(f"  Trenuję finalny SGD/Ridge [α={val_str}] ({N_EPOK} epok)...")
    finalny_model = trenuj_sgd(X_train, y_train.values, n_epok=N_EPOK, alpha=float(val_str))
    # Zachowaj też Ridge do krzywej uczenia
    ridge_alpha = float(val_str)
elif najlepszy_nazwa.startswith("SVR"):
    print(f"  Trenuję finalny SVR [C={val_str}]...")
    finalny_model = trenuj_svr(X_train[idx_svr], y_train.iloc[idx_svr], C=float(val_str))
    ridge_alpha = 0.0001
else:
    print(f"  Buduję finalny RandomForest [n={val_str}]...")
    finalny_model = trenuj_rf(X_train, y_train, n_drzew=int(val_str))
    ridge_alpha = 0.0001

y_pred_final = np.clip(finalny_model.predict(X_test), 1.0, 5.0)
acc_final    = proximity_accuracy(y_test.values, y_pred_final)
mae_final    = mae_score(y_test.values, y_pred_final)
rmse_final   = float(np.sqrt(np.mean((y_pred_final - y_test.values)**2)))

print(f"\n  Finalny model:         {najlepszy_nazwa} [{najlepszy_param}]")
print(f"  Proximity Accuracy:    {acc_final*100:.2f}%")
print(f"  MAE:                   {mae_final:.4f} gwiazdki")
print(f"  RMSE:                  {rmse_final:.4f} gwiazdki")

# ZAPIS FINALNEGO MODELU (DODANO)
print(f"\n--- ZAPIS FINALNEGO MODELU ({najlepszy_nazwa}) ---")
joblib.dump(finalny_model, "finalny_model_po_tuningu.joblib")
print("Zapisano pomyślnie: finalny_model_po_tuningu.joblib")

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
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Analiza błędów per kategoria oceny", fontsize=13, fontweight='bold')
sns.barplot(data=stats, x='ocena_int', y='sredni_blad', ax=axes[0], palette='Reds_d')
axes[0].set_title("Średni błąd bezwzględny")
axes[0].set_xlabel("Rzeczywista ocena")
axes[0].set_ylabel("MAE (gwiazdki)")
sns.barplot(data=stats, x='ocena_int', y='skutecznosc_avg', ax=axes[1], palette='Greens_d')
axes[1].set_title("Średnia skuteczność")
axes[1].set_xlabel("Rzeczywista ocena")
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
print(f"  SGD/Ridge [α={ridge_alpha}], 10 punktów × 3 foldy = 30 kroków\n")

N_PUNKTOW = 10
N_FOLDOW  = 3
udzialy   = np.linspace(0.05, 1.0, N_PUNKTOW)
kf        = KFold(n_splits=N_FOLDOW, shuffle=True, random_state=42)

train_sizes_abs     = []
train_maes_wszystkie = []
test_maes_wszystkie  = []

total_k8 = N_PUNKTOW * N_FOLDOW
krok_k8  = 0
t_k8     = time.time()

for ui, udzial in enumerate(udzialy):
    n_prob = int(X_train.shape[0] * udzial)
    idx_u  = np.random.default_rng(ui).choice(X_train.shape[0], n_prob, replace=False)
    X_u    = X_train[idx_u]
    y_u    = y_train.values[idx_u]

    fold_train = []
    fold_test  = []

    for fi, (tr_idx, val_idx) in enumerate(kf.split(X_u), 1):
        krok_k8 += 1
        opis = f"punkt {ui+1:>2}/{N_PUNKTOW} ({udzial*100:.0f}% danych, {n_prob:,} wierszy)  fold {fi}/{N_FOLDOW}"
        pasek_postepu(krok_k8, total_k8, opis, t_k8)

        m = SGDRegressor(loss='squared_error', penalty='l2', alpha=ridge_alpha,
                         max_iter=50, random_state=42, tol=1e-3)
        m.fit(X_u[tr_idx], y_u[tr_idx])
        fold_train.append(mae_score(y_u[tr_idx], m.predict(X_u[tr_idx])))
        fold_test.append(mae_score(y_u[val_idx], m.predict(X_u[val_idx])))

    train_sizes_abs.append(n_prob)
    train_maes_wszystkie.append(fold_train)
    test_maes_wszystkie.append(fold_test)

nowa_linia()
print(f"  Krzywa zakończona (łączny czas: {time.time()-t_k8:.1f}s)")

train_mae_avg = np.array([np.mean(v) for v in train_maes_wszystkie])
train_mae_std = np.array([np.std(v)  for v in train_maes_wszystkie])
test_mae_avg  = np.array([np.mean(v) for v in test_maes_wszystkie])
test_mae_std  = np.array([np.std(v)  for v in test_maes_wszystkie])
train_acc_avg = 1 - train_mae_avg / SKALA
test_acc_avg  = 1 - test_mae_avg  / SKALA

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Krzywa uczenia się", fontsize=13, fontweight='bold')

axes[0].plot(train_sizes_abs, train_mae_avg, 'o-', color='steelblue', label='Treningowy')
axes[0].plot(train_sizes_abs, test_mae_avg,  's-', color='coral',     label='Walidacyjny')
axes[0].fill_between(train_sizes_abs,
    train_mae_avg - train_mae_std, train_mae_avg + train_mae_std, alpha=0.15, color='steelblue')
axes[0].fill_between(train_sizes_abs,
    test_mae_avg - test_mae_std,   test_mae_avg + test_mae_std,   alpha=0.15, color='coral')
axes[0].set_title("MAE vs liczba próbek treningowych")
axes[0].set_xlabel("Liczba próbek treningowych")
axes[0].set_ylabel("MAE (gwiazdki)")
axes[0].legend()

axes[1].plot(train_sizes_abs, train_acc_avg, 'o-', color='steelblue', label='Treningowy')
axes[1].plot(train_sizes_abs, test_acc_avg,  's-', color='coral',     label='Walidacyjny')
axes[1].fill_between(train_sizes_abs,
    1-(train_mae_avg+train_mae_std)/SKALA, 1-(train_mae_avg-train_mae_std)/SKALA,
    alpha=0.15, color='steelblue')
axes[1].fill_between(train_sizes_abs,
    1-(test_mae_avg+test_mae_std)/SKALA,   1-(test_mae_avg-test_mae_std)/SKALA,
    alpha=0.15, color='coral')
axes[1].set_title("Proximity Accuracy vs liczba próbek")
axes[1].set_xlabel("Liczba próbek treningowych")
axes[1].set_ylabel("Proximity Accuracy")
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
axes[1].legend()

plt.tight_layout()
plt.show()

print("\n--- MILESTONE 4 ZAKOŃCZONY POMYŚLNIE ---")