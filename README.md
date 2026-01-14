# Turkish-News-Classification-NLP
A comparative NLP project analyzing Turkish news texts using BERT, Word2Vec, and TF-IDF models.
# Turkish News Classification with NLP Models 🇹🇷

This project classifies Turkish news articles into categories (Economy, Sports, Technology) using a comparative approach with traditional ML and Deep Learning techniques.

## 🚀 Features
* **Comparison:** Bag-of-Words, TF-IDF, Word2Vec, and BERT.
* **Dataset:** Custom Turkish news dataset using `VeriSeti_Ali.csv`.
* **Accuracy:** Uses Logistic Regression for final classification logic.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, PyTorch, Transformers, Gensim

## 📊 How It Works
1.  **Preprocessing:** Cleans text (lowercasing, punctuation removal).
2.  **Model Selection:** Randomly selects an embedding technique (BERT, Word2Vec, etc.).
3.  **Classification:** Uses Logistic Regression to predict the category.

## 💻 Usage
```bash
pip install -r requirements.txt
python main.py
