# 🧠 Duplicate Question Pair Detection

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-blue" />
  <img src="https://img.shields.io/badge/Frontend-Streamlit-red" />
  <img src="https://img.shields.io/badge/NLP-Text%20Processing-green" />
</p>

---

## 🚀 Overview

This project predicts whether two questions are **duplicates or not** using Natural Language Processing and Machine Learning.

It mimics real-world systems like **Quora Question Pair similarity detection**.

---

## 🎯 Features

✅ Text preprocessing (cleaning, normalization)
✅ Token-based similarity features
✅ Fuzzy matching techniques
✅ Bag-of-Words vectorization
✅ Machine Learning prediction
✅ Interactive UI using Streamlit

---

## 🧱 Project Structure

```
Question-Pair-Model/
│── app.py                # Streamlit app
│── helper.py            # Feature engineering & preprocessing
│── model.pkl            # Trained ML model
│── cv.pkl               # CountVectorizer
│── requirements.txt     # Dependencies
```

---

## ⚙️ Tech Stack

| Category   | Tools Used       |
| ---------- | ---------------- |
| Language   | Python           |
| ML Library | Scikit-learn     |
| NLP        | NLTK, FuzzyWuzzy |
| UI         | Streamlit        |

---


## ▶️ Run Locally

```bash
# Clone repo
git clone https://github.com/your-username/Beginner-Python-Projects.git

# Go inside project
cd Beginner-Python-Projects/DUPLICATE-QUESTION-PAIRS

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

---

## 🧪 Example

**Input:**

* Q1: How many hours should we sleep in a week?
* Q2: For how many hours we should try to get sleep in a week?

**Output:**
👉 Duplicate ✅

---

## 📌 Future Improvements

* Use Deep Learning (LSTM / BERT)
* Deploy on cloud


---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## ⭐ Show Your Support

If you like this project, give it a ⭐ on GitHub!
