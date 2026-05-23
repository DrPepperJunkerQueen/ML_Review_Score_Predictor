# 🌟 ML Review Score Predictor

Welcome to the **ML Review Score Predictor**! This repository hosts a Chrome extension and a local Python-based API designed to predict star ratings from any selected text using a custom Machine Learning model.

As shown in the project structure within `image_ca904d.png`, the repository is cleanly divided into a web extension and a local backend processing server.

---

## 🚀 Features

* **Context Menu Integration**: Simply highlight text in your browser, right-click, and select "Wylicz ocenę dla tego tekstu" (Calculate rating for this text) to instantly get a prediction.


* **Customizable Display**: Configure the extension to display results in either a standard 1-5 star format (⭐) or a 10-point scale depending on your preferences.


* **Local Processing**: The extension securely sends the text to a locally hosted Flask server, keeping your data entirely on your machine.


* **Smart Edge-Case Handling**: For extremely negative texts scoring below 1.25 on the raw scale, the extension intelligently scales the score down to a 0/10 with an added explanatory note.



---

## 📁 Repository Structure

The project is organized into two primary directories as seen in `image_ca904d.png`:

### 1. `api_backend/`

This folder contains the machine learning pipeline and the Flask server.

* **`api_serwer.py`**: The main Flask application that listens on port 5000 and serves the `/predict` endpoint.


* **Milestone Scripts**:
* `milestone_1.py`: Fetches and processes raw data from the Amazon Reviews 2023 dataset.


* `milestone_2.py`: Cleans HTML tags, balances the dataset, and performs TF-IDF vectorization.


* `milestone_3.py`: Trains an initial Ridge regression model for baseline predictions.


* `milestone_4.py`: Compares advanced models (SGD, SVR, Random Forest), handles hyperparameter tuning, and exports the ultimate champion model.




* **Sample Reviews (`.txt`)**: Included in this folder are several text files containing sample reviews for quick local testing of the models. As a neat detail, the names of these text files actually correspond to the names of the original review authors!
* ⚠️ **Note on Large Files**: Dataset files (`.parquet`) and trained model files (`.joblib`, like `milestone3_vectorizer.joblib` and `finalny_model_po_tuningu.joblib`) are excluded from the public repository via `.gitignore` because they are too large. You must run the milestone scripts sequentially to generate them locally!



### 2. `chrome_extension/`

This folder houses the frontend user interface.

* **`manifest.json`**: The Manifest V3 configuration file granting permissions for context menus, storage, and active tabs.


* **`background.js`**: The service worker that manages the context menu clicks, user settings retrieval, and API `fetch` requests.


* **`options.html`** & **`options.js**`**: The interface where you can toggle your preferred rating scale (points vs. stars).



---

## 🧠 How the Machine Learning Works

The AI behind this extension was built in four distinct phases:

1. **Data Collection**: Over 500,000 reviews from Video Games and Movies & TV were compiled and prepared using Hugging Face datasets.


2. **Data Preprocessing**: Reviews with fewer than 5 words were dropped, class imbalances were fixed via undersampling, and the text was transformed into numerical data using TF-IDF (Term Frequency-Inverse Document Frequency).


3. **Baseline Training**: A baseline Ridge Regression model was trained and evaluated for Mean Absolute Error (MAE) and Proximity Accuracy.


4. **Model Optimization**: Multiple architectures (SGD/Ridge, SVR, Random Forest) battled it out. The script dynamically tuned hyperparameters and saved the top performer as `finalny_model_po_tuningu.joblib`.



---

## 🛠️ Installation & Setup

### Step 1: Generate Models & Start the Backend Server

1. Navigate to the `api_backend/` directory.
2. Because `.parquet` and `.joblib` files are ignored by git, you **must** run the milestone scripts (`milestone_1.py` through `milestone_4.py`) first to download the data and train the models.
3. Once `milestone3_vectorizer.joblib` and `finalny_model_po_tuningu.joblib` are successfully generated in the folder, run the Flask server:


```bash
python api_serwer.py
```


4. You should see a message confirming that the server is listening on port 5000.

### Step 2: Install the Chrome Extension
1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Toggle **Developer mode** on in the top right corner.
3. Click **Load unpacked** and select the `chrome_extension/` folder from this repository.
4. The extension (listed under its original manifest name "ML Review Predictor v1.1") should now be active.

### Step 3: Test it out!
1. Highlight any text on a webpage.
2. Right-click and choose **Wylicz ocenę dla tego tekstu**.
3. Make sure the Python server is running, and enjoy your AI-generated rating! 🎯
