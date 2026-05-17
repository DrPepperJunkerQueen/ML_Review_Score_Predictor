import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import sys

app = Flask(__name__)
CORS(app) 

# Magiczna linijka: pobiera absolutną ścieżkę do folderu, w którym leży ten skrypt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sklejamy ścieżkę folderu z nazwami plików
sciezka_vec = os.path.join(BASE_DIR, "milestone3_vectorizer.joblib")
sciezka_mod = os.path.join(BASE_DIR, "finalny_model_po_tuningu.joblib")

print("=========================================")
print(f"Szukam wektoryzatora w: {sciezka_vec}")
print(f"Szukam modelu w: {sciezka_mod}")
print("=========================================")

# Próbujemy załadować. Jeśli plików nie ma, program się zatrzyma i rzuci błędem.
try:
    vectorizer = joblib.load(sciezka_vec)
    model = joblib.load(sciezka_mod)
    print("✓ Pliki wczytane pomyślnie!")
    print("✓ Serwer nasłuchuje na porcie 5000.")
except FileNotFoundError as e:
    print(f"\n[!] BŁĄD KRYTYCZNY: Nie znaleziono plików modelu!")
    print(f"Upewnij się, że wygenerowałeś modele i leżą one w folderze: {BASE_DIR}")
    sys.exit(1) # Zatrzymujemy serwer
except Exception as e:
    print(f"\n[!] BŁĄD KRYTYCZNY przy ładowaniu plików: {e}")
    sys.exit(1)


@app.route('/predict', methods=['POST'])
def predict():
    dane = request.get_json()
    
    if not dane or 'text' not in dane:
        return jsonify({'error': 'Brak tekstu do analizy'}), 400
    
    tekst = dane['text']
    
    if len(tekst.strip()) < 5:
        return jsonify({'message': 'Zaznaczony tekst jest za krótki (min. 5 znaków).'})
        
    try:
        wektor = vectorizer.transform([tekst])
        wynik_raw = model.predict(wektor)[0]
        wynik = float(np.clip(wynik_raw, 1.0, 5.0))
        
        return jsonify({'rating': round(wynik, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)