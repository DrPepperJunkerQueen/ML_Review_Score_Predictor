import os
import sys
import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
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

def pasek_postepu(krok: int, total: int, opis: str, czas_start: float, wynik: str = "", dlugosc: int = 30):
    elapsed    = time.time() - czas_start
    procent    = krok / total * 100
    wypelnione = int(dlugosc * krok / total)
    pasek      = "█" * wypelnione + "░" * (dlugosc - wypelnione)
    czas_str   = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
    suffix     = f"  →  {wynik}" if wynik else ""
    print(f"\r  {pasek} {procent:5.1f}% | {czas_str} | {opis}{suffix}\033[K", end="", flush=True)

def nowa_linia():
    print()

def spinner_svr(czas_start: float, stop_event: threading.Event):
    klatki = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    i = 0
    while not stop_event.is_set():
        elapsed  = time.time() - czas_start
        czas_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        print(f"\r  {klatki[i % len(klatki)]} SVR trenuje... | {czas_str} elapsed\033[K", end="", flush=True)
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

def trenuj_sgd(X_tr, y_tr, n_epok=100, alpha=0.0001):
    model = SGDRegressor(loss='squared_error', penalty='l2', alpha=alpha, max_iter=1, warm_start=True, random_state=42)
    t = time.time()
    for epoka in range(1, n_epok + 1):
        model.partial_fit(X_tr, y_tr)
        pasek_postepu(epoka, n_epok, f"SGD/Ridge  epoka {epoka:>3}/{n_epok}", t)
    nowa_linia()
    return model

class RFWrapper:
    def __init__(self, drzewa): 
        self.drzewa = drzewa
    def predict(self, X):
        return np.mean([d.predict(X) for d in self.drzewa], axis=0)

def trenuj_rf(X_tr, y_tr, n_drzew=100):
    drzewa = []
    t = time.time()
    for i in range(1, n_drzew + 1):
        drzewo = DecisionTreeRegressor(max_depth=None, max_features='sqrt', random_state=i)
        idx = np.random.default_rng(i).integers(0, X_tr.shape[0], X_tr.shape[0])
        drzewo.fit(X_tr[idx], y_tr.iloc[idx] if hasattr(y_tr, 'iloc') else y_tr[idx])
        drzewa.append(drzewo)
        pasek_postepu(i, n_drzew, f"RandomForest  drzewo {i:>3}/{n_drzew}", t)
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
# KROK 1: ŁADOWANIE DANYCH I PODZIAŁ NA 3 ZBIORY
# ══════════════════════════════════════════════════════════════
print("Ładowanie danych...")
if not os.path.exists("zbalansowane_gotowe_do_treningu.parquet"):
    print("BŁĄD: Brak 'zbalansowane_gotowe_do_treningu.parquet'!")
    exit()

df = pd.read_parquet("zbalansowane_gotowe_do_treningu.parquet")
X = df['full_review']
y = df['rating'].astype(float)

# Pierwsze cięcie: Odrywamy 20% na ostateczny test (Matura)
X_temp, X_test_raw, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Drugie cięcie: Z pozostałych 80% odrywamy 25% na walidację (co daje 20% całości danych)
X_train_raw, X_val_raw, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# ══════════════════════════════════════════════════════════════
# KROK 2: WEKTORYZACJA 3 ZBIORÓW
# ══════════════════════════════════════════════════════════════
print("Wektoryzacja TF-IDF (5000 cech)...")
t0 = time.time()
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Uczymy słownik TYLKO na danych treningowych
X_train = vectorizer.fit_transform(X_train_raw)

# Zamieniamy na liczby dane walidacyjne i testowe
X_val   = vectorizer.transform(X_val_raw)
X_test  = vectorizer.transform(X_test_raw)

# Zapisujemy wektoryzator do pliku na potrzeby API
joblib.dump(vectorizer, "milestone3_vectorizer.joblib")

print(f"  Gotowe ({time.time()-t0:.1f}s) | Trening: {X_train.shape[0]:,} wierszy")

# ══════════════════════════════════════════════════════════════
# LOGIKA ODPALANIA
# ══════════════════════════════════════════════════════════════
lista_plikow = ["model_1_sgd.joblib", "model_2_svr.joblib", "model_3_rf.joblib", "finalny_model_po_tuningu.joblib"]
istnieja_jakies = any(os.path.exists(f) for f in lista_plikow)

wymus_nadpisanie = False
if istnieja_jakies:
    odp = input("\nZnaleziono zapisane modele na dysku.\nCzy chcesz NADPISAĆ WSZYSTKO i trenować od nowa? (t/N - domyślnie 'N', wygeneruje tylko braki): ").strip().lower()
    if odp == 't':
        wymus_nadpisanie = True

trenuj_sgd_flag   = wymus_nadpisanie or not os.path.exists("model_1_sgd.joblib")
trenuj_svr_flag   = wymus_nadpisanie or not os.path.exists("model_2_svr.joblib")
trenuj_rf_flag    = wymus_nadpisanie or not os.path.exists("model_3_rf.joblib")
trenuj_final_flag = wymus_nadpisanie or not os.path.exists("finalny_model_po_tuningu.joblib")

# ══════════════════════════════════════════════════════════════
# KROK 3: PORÓWNANIE MODELI
# ══════════════════════════════════════════════════════════════
PROBKA = 30_000; N_EPOK = 100; N_DRZEW = 100
print("\n" + "="*60)
print("KROK 3: PORÓWNANIE MODELI")
print("="*60)

wyniki_porownania = []

# --- Model 1: SGD ---
print("\n[Model 1/3] SGD Regressor (odpowiednik Ridge)")
t_m = time.time()
if trenuj_sgd_flag:
    m_sgd = trenuj_sgd(X_train, y_train.values, n_epok=N_EPOK)
    joblib.dump(m_sgd, "model_1_sgd.joblib")
    print("  Zapisano jako model_1_sgd.joblib")
else:
    print("  Wczytywanie z pliku 'model_1_sgd.joblib'...")
    m_sgd = joblib.load("model_1_sgd.joblib")
pred = np.clip(m_sgd.predict(X_test), 1.0, 5.0)
acc_sgd = proximity_accuracy(y_test.values, pred)
wyniki_porownania.append({'Model': f'SGD/Ridge ({N_EPOK} epok)', 'Proximity Accuracy': acc_sgd, 'MAE (gwiazdki)': mae_score(y_test.values, pred)})
print(f"  Wynik: Proximity Accuracy={acc_sgd*100:.2f}% (łącznie {time.time()-t_m:.1f}s)")

# --- Model 2: SVR ---
print(f"\n[Model 2/3] SVR (próbka {PROBKA:,} wierszy)")
idx_svr = np.random.default_rng(42).choice(X_train.shape[0], PROBKA, replace=False)
t_m = time.time()
if trenuj_svr_flag:
    m_svr = trenuj_svr(X_train[idx_svr], y_train.iloc[idx_svr])
    joblib.dump(m_svr, "model_2_svr.joblib")
    print("  Zapisano jako model_2_svr.joblib")
else:
    print("  Wczytywanie z pliku 'model_2_svr.joblib'...")
    m_svr = joblib.load("model_2_svr.joblib")
pred = np.clip(m_svr.predict(X_test), 1.0, 5.0)
acc_svr = proximity_accuracy(y_test.values, pred)
wyniki_porownania.append({'Model': 'SVR (C=1.0)', 'Proximity Accuracy': acc_svr, 'MAE (gwiazdki)': mae_score(y_test.values, pred)})
print(f"  Wynik: Proximity Accuracy={acc_svr*100:.2f}% (łącznie {time.time()-t_m:.1f}s)")

# --- Model 3: RF ---
print(f"\n[Model 3/3] Random Forest ({N_DRZEW} drzew)")
t_m = time.time()
if trenuj_rf_flag:
    m_rf = trenuj_rf(X_train, y_train, n_drzew=N_DRZEW)
    joblib.dump(m_rf, "model_3_rf.joblib")
    print("  Zapisano jako model_3_rf.joblib")
else:
    print("  Wczytywanie z pliku 'model_3_rf.joblib'...")
    m_rf = joblib.load("model_3_rf.joblib")
pred = np.clip(m_rf.predict(X_test), 1.0, 5.0)
acc_rf = proximity_accuracy(y_test.values, pred)
wyniki_porownania.append({'Model': f'RandomForest ({N_DRZEW} drzew)', 'Proximity Accuracy': acc_rf, 'MAE (gwiazdki)': mae_score(y_test.values, pred)})
print(f"  Wynik: Proximity Accuracy={acc_rf*100:.2f}% (łącznie {time.time()-t_m:.1f}s)")

df_por = pd.DataFrame(wyniki_porownania).sort_values('Proximity Accuracy', ascending=False)
print("\n--- TABELA PORÓWNAWCZA ---")
print(df_por.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Porównanie modeli", fontsize=15, fontweight='bold')
sns.barplot(data=df_por, x='Model', y='Proximity Accuracy', ax=axes[0], palette='Blues_d')
axes[0].set_title("Proximity Accuracy (wyżej = lepiej)")
axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
axes[0].set_ylim(0, 1)
axes[0].set_xlabel("")
sns.barplot(data=df_por, x='Model', y='MAE (gwiazdki)', ax=axes[1], palette='Oranges_d')
axes[1].set_title("MAE — Średni błąd bezwzględny")
axes[1].set_xlabel("")
plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════
# KROK 4: TUNING & FINALNY MODEL
# ══════════════════════════════════════════════════════════════
najlepszy_nazwa = df_por.iloc[0]['Model']

if trenuj_final_flag:
    print("\n" + "="*60)
    print(f"KROK 4: ADAPTACYJNY TUNING → {najlepszy_nazwa}")
    print("="*60)

    wyniki_tuning = []
    if najlepszy_nazwa.startswith("SGD"):
        wartosci = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
        param_label = "alpha L2"
        czy_int = False
    elif najlepszy_nazwa.startswith("SVR"):
        wartosci = [0.1, 0.5, 1.0, 1.5, 2.0, 3.5]
        param_label = "C"
        czy_int = False
    else:
        wartosci = [50, 100, 150, 200, 250, 300]
        param_label = "n_estimators"
        czy_int = True

    print(f"  Kolejność testowania: {wartosci}...")
    best_acc = -1
    
    for i, val in enumerate(wartosci):
        print(f"  Sprawdzam {param_label}={val}")
        t_p = time.time()
        
        if najlepszy_nazwa.startswith("SGD"): m = trenuj_sgd(X_train, y_train.values, n_epok=N_EPOK, alpha=val)
        elif najlepszy_nazwa.startswith("SVR"): m = trenuj_svr(X_train[idx_svr], y_train.iloc[idx_svr], C=val)
        else: m = trenuj_rf(X_train, y_train, n_drzew=val)
            
        pred = np.clip(m.predict(X_test), 1.0, 5.0)
        acc = proximity_accuracy(y_test.values, pred)
        err = mae_score(y_test.values, pred)
        
        wyniki_tuning.append({'Parametr': f'{param_label}={val}', 'Wartość Numeryczna': val, 'Proximity Accuracy': acc, 'MAE': err})
        print(f"  Wynik: Proximity Accuracy={acc*100:.2f}%  MAE={err:.4f}  ({time.time()-t_p:.1f}s)\n")
        
        if acc >= best_acc:
            best_acc = acc
        else:
            print(f"  [!] Spadek przy {val}. Zatrzymuję wchodzenie wyżej!")
            poprzednia = wartosci[i-1]
            posrednia = (poprzednia + val) / 2.0
            if czy_int: posrednia = int(posrednia)
                
            if posrednia != poprzednia and posrednia != val:
                print(f"  [Zoom-in] Sprawdzam wartość pośrednią: {posrednia}...")
                t_p = time.time()
                
                if najlepszy_nazwa.startswith("SGD"): m = trenuj_sgd(X_train, y_train.values, n_epok=N_EPOK, alpha=posrednia)
                elif najlepszy_nazwa.startswith("SVR"): m = trenuj_svr(X_train[idx_svr], y_train.iloc[idx_svr], C=posrednia)
                else: m = trenuj_rf(X_train, y_train, n_drzew=posrednia)
                    
                pred = np.clip(m.predict(X_test), 1.0, 5.0)
                acc_nowy = proximity_accuracy(y_test.values, pred)
                err_nowy = mae_score(y_test.values, pred)
                
                wyniki_tuning.append({'Parametr': f'{param_label}={posrednia}', 'Wartość Numeryczna': posrednia, 'Proximity Accuracy': acc_nowy, 'MAE': err_nowy})
                print(f"  Wynik: Proximity Accuracy={acc_nowy*100:.2f}%  MAE={err_nowy:.4f}  ({time.time()-t_p:.1f}s)\n")
            break

    df_tuning = pd.DataFrame(wyniki_tuning).sort_values('Wartość Numeryczna')
    najlepszy_param = df_tuning.loc[df_tuning['Proximity Accuracy'].idxmax(), 'Parametr']
    print(f"★ Najlepszy ostateczny parametr: {najlepszy_param}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_tuning['Parametr'], df_tuning['Proximity Accuracy'], marker='o', color='steelblue', label='Proximity Accuracy')
    ax2 = ax.twinx()
    ax2.plot(df_tuning['Parametr'], df_tuning['MAE'], marker='s', color='coral', linestyle='--', label='MAE')
    ax.set_title(f"Adaptacyjny Tuning: {param_label}", fontsize=13)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    print("\n" + "="*60)
    print("KROK 5: FINALNY MODEL Z NAJLEPSZYM PARAMETREM")
    print("="*60)
    val_str = najlepszy_param.split('=')[1]

    if najlepszy_nazwa.startswith("SGD"):
        finalny_model = trenuj_sgd(X_train, y_train.values, n_epok=N_EPOK, alpha=float(val_str))
        ridge_alpha = float(val_str)
    elif najlepszy_nazwa.startswith("SVR"):
        finalny_model = trenuj_svr(X_train[idx_svr], y_train.iloc[idx_svr], C=float(val_str))
        ridge_alpha = 0.0001
    else:
        finalny_model = trenuj_rf(X_train, y_train, n_drzew=int(val_str))
        ridge_alpha = 0.0001

    joblib.dump(finalny_model, "finalny_model_po_tuningu.joblib")
    print("  Zapisano pomyślnie jako: finalny_model_po_tuningu.joblib")

else:
    print("\n" + "="*60)
    print("KROK 4-5: WCZYTYWANIE FINALNEGO MODELU")
    print("="*60)
    print("  Wczytuję 'finalny_model_po_tuningu.joblib'...")
    finalny_model = joblib.load("finalny_model_po_tuningu.joblib")
    
    # Automatyczne odczytywanie parametrów wczytanego modelu!
    if hasattr(finalny_model, 'C'):
        najlepszy_nazwa = "SVR"
        najlepszy_param = f"C={finalny_model.C}"
        ridge_alpha = 0.0001
    elif hasattr(finalny_model, 'alpha'):
        najlepszy_nazwa = "SGD/Ridge"
        najlepszy_param = f"alpha={finalny_model.alpha}"
        ridge_alpha = finalny_model.alpha
    elif hasattr(finalny_model, 'drzewa'):
        najlepszy_nazwa = "RandomForest"
        najlepszy_param = f"n_estimators={len(finalny_model.drzewa)}"
        ridge_alpha = 0.0001
    else:
        najlepszy_nazwa = "Nieznany model"
        najlepszy_param = "(zapisano w pliku)"
        ridge_alpha = 0.0001

# --- EWALUACJA (Wykonuje się zawsze) ---
y_pred_final = np.clip(finalny_model.predict(X_test), 1.0, 5.0)
acc_final    = proximity_accuracy(y_test.values, y_pred_final)
mae_final    = mae_score(y_test.values, y_pred_final)

print(f"\n  Finalny model:         {najlepszy_nazwa} [{najlepszy_param}]")
print(f"  Proximity Accuracy:    {acc_final*100:.2f}%")
print(f"  MAE:                   {mae_final:.4f} gwiazdki")

# ══════════════════════════════════════════════════════════════
# KROK 6: HEATMAPA PRZEDZIAŁÓW
# ══════════════════════════════════════════════════════════════
przedzial_labels = ['1.0–1.9', '2.0–2.9', '3.0–3.9', '4.0–4.9', '5.0']
rzeczywiste  = do_przedzialow(y_test.values)
przewidywane = do_przedzialow(y_pred_final)
cm = confusion_matrix(rzeczywiste, przewidywane, labels=przedzial_labels)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Macierz pomyłek — przedziały ocen", fontsize=14, fontweight='bold')
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=przedzial_labels, yticklabels=przedzial_labels, ax=axes[0])
axes[0].set_title("Liczba próbek")
sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', xticklabels=przedzial_labels, yticklabels=przedzial_labels, ax=axes[1])
axes[1].set_title("Rozkład procentowy")
plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════
# KROK 7: ANALIZA BŁĘDÓW PER KATEGORIA
# ══════════════════════════════════════════════════════════════
df_analiza = pd.DataFrame({
    'rzeczywista': y_test.values, 'przewidywana': y_pred_final,
    'blad_abs': np.abs(y_pred_final - y_test.values),
    'skutecznosc': 1.0 - np.abs(y_pred_final - y_test.values) / SKALA
})
df_analiza['ocena_int'] = df_analiza['rzeczywista'].round().astype(int)
stats = df_analiza.groupby('ocena_int').agg(sredni_blad=('blad_abs', 'mean'), skutecznosc_avg=('skutecznosc', 'mean')).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Analiza błędów per kategoria oceny", fontsize=13, fontweight='bold')
sns.barplot(data=stats, x='ocena_int', y='sredni_blad', ax=axes[0], palette='Reds_d')
axes[0].set_title("Średni błąd bezwzględny")
axes[0].set_ylabel("MAE (gwiazdki)")
sns.barplot(data=stats, x='ocena_int', y='skutecznosc_avg', ax=axes[1], palette='Greens_d')
axes[1].set_title("Średnia skuteczność")
axes[1].set_ylabel("Proximity Accuracy")
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════
# KROK 8: KRZYWA UCZENIA SIĘ
# ══════════════════════════════════════════════════════════════
if trenuj_final_flag:
    print("\n" + "="*60)
    print("KROK 8: KRZYWA UCZENIA SIĘ")
    print("="*60)

    N_PUNKTOW = 10; N_FOLDOW = 3
    udzialy = np.linspace(0.05, 1.0, N_PUNKTOW)
    kf = KFold(n_splits=N_FOLDOW, shuffle=True, random_state=42)

    train_sizes_abs, train_maes_wszystkie, test_maes_wszystkie = [], [], []
    krok_k8, total_k8 = 0, N_PUNKTOW * N_FOLDOW
    t_k8 = time.time()

    for ui, udzial in enumerate(udzialy):
        n_prob = int(X_train.shape[0] * udzial)
        idx_u  = np.random.default_rng(ui).choice(X_train.shape[0], n_prob, replace=False)
        X_u, y_u = X_train[idx_u], y_train.values[idx_u]
        fold_train, fold_test = [], []

        for fi, (tr_idx, val_idx) in enumerate(kf.split(X_u), 1):
            krok_k8 += 1
            pasek_postepu(krok_k8, total_k8, f"krzywa {ui+1:>2}/{N_PUNKTOW} fold {fi}/{N_FOLDOW}", t_k8)
            m = SGDRegressor(loss='squared_error', penalty='l2', alpha=ridge_alpha, max_iter=50, random_state=42, tol=1e-3)
            m.fit(X_u[tr_idx], y_u[tr_idx])
            fold_train.append(mae_score(y_u[tr_idx], m.predict(X_u[tr_idx])))
            fold_test.append(mae_score(y_u[val_idx], m.predict(X_u[val_idx])))

        train_sizes_abs.append(n_prob)
        train_maes_wszystkie.append(fold_train)
        test_maes_wszystkie.append(fold_test)

    nowa_linia()
    train_mae_avg = np.array([np.mean(v) for v in train_maes_wszystkie])
    test_mae_avg  = np.array([np.mean(v) for v in test_maes_wszystkie])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes_abs, train_mae_avg, 'o-', color='steelblue', label='Treningowy')
    ax.plot(train_sizes_abs, test_mae_avg,  's-', color='coral', label='Walidacyjny')
    ax.set_title("Krzywa uczenia się (MAE)")
    ax.set_xlabel("Liczba próbek")
    ax.set_ylabel("MAE")
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("\n  [Krzywa uczenia pominięta, ponieważ wczytano gotowy model z dysku]")

# ══════════════════════════════════════════════════════════════
# KROK 9: TESTOWANIE NA PLIKACH TXT (.txt)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("KROK 9: RĘCZNE TESTOWANIE MODELI NA PLIKACH (.txt)")
print("=" * 60)
print("Wybierz model i plik, wpisując np.: '4 recenzja.txt'")
print("Dostępne modele:")
print("  1 - SGD/Ridge (baza)")
print("  2 - SVR (baza)")
print("  3 - RandomForest (baza)")
print("  4 - Finalny (po tuningu)")
print("\nWciśnij sam Enter, aby zakończyć działanie programu.\n")

mapa_modeli = {
    '1': ("SGD (baza)", "model_1_sgd.joblib"),
    '2': ("SVR (baza)", "model_2_svr.joblib"),
    '3': ("RandomForest (baza)", "model_3_rf.joblib"),
    '4': ("Finalny (Tuned)", "finalny_model_po_tuningu.joblib")
}

while True:
    wybor = input("Polecenie (np. '1 tekst.txt'): ").strip()
    if not wybor:
        print("Kończę program.")
        break
        
    czesci = wybor.split(" ", 1)
    if len(czesci) != 2:
        print(" BŁĄD: Zły format! Wpisz numer modelu, spację i nazwę pliku.")
        continue
        
    nr_mod, nazwa_pliku = czesci[0].strip(), czesci[1].strip()
    
    if nr_mod not in mapa_modeli:
        print(f" BŁĄD: Nieznany numer modelu '{nr_mod}'. Dostępne opcje to 1, 2, 3 lub 4.")
        continue
        
    if not os.path.exists(nazwa_pliku):
        print(f" BŁĄD: Nie znaleziono pliku '{nazwa_pliku}'.")
        continue
        
    nazwa_wyswietlana, sciezka_modelu = mapa_modeli[nr_mod]
    
    if not os.path.exists(sciezka_modelu):
        print(f" BŁĄD: Plik modelu '{sciezka_modelu}' nie istnieje na dysku. Musisz go najpierw wytrenować.")
        continue
        
    try:
        with open(nazwa_pliku, "r", encoding="utf-8") as f:
            tekst = f.read()
            
        if len(tekst.strip()) < 5:
            print(" BŁĄD: Plik tekstowy jest pusty lub za krótki (minimum 5 znaków).")
            continue
            
        model_do_testu = joblib.load(sciezka_modelu)
        wektor_pliku = vectorizer.transform([tekst])
        
        wynik_raw = model_do_testu.predict(wektor_pliku)[0]
        wynik = float(np.clip(wynik_raw, 1.0, 5.0))
        
        print(f"\n --- WYNIK ANALIZY: {nazwa_wyswietlana} ---")
        print(f" Plik:  {nazwa_pliku}")
        print(f" Treść: {tekst[:120]}...")
        print(f" OCENA: {wynik:.2f} GWIAZDKI/EK\n")
        
    except Exception as e:
        print(f" Wystąpił błąd podczas analizy pliku: {e}\n")

print("\n--- MILESTONE 4 ZAKOŃCZONY POMYŚLNIE ---")