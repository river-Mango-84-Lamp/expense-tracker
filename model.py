"""ML part: expense text se category predict karna + agle mahine ka kharcha predict karna."""
import random

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline

CATEGORIES = ["Food", "Travel", "Shopping", "Bills", "Education", "Entertainment", "Health"]

# Har category ke liye kuch common words. Aap apne hisaab se aur add kar sakte ho.
KEYWORDS = {
    "Food": ["zomato", "swiggy", "pizza", "burger", "biryani", "chai", "coffee", "canteen",
             "lunch", "dinner", "breakfast", "maggi", "samosa", "juice", "groceries",
             "vegetables", "milk", "dominos", "kfc", "mess", "snacks", "ice cream"],
    "Travel": ["auto", "uber", "ola", "rapido", "bus", "metro", "petrol", "diesel",
               "train ticket", "cab", "taxi", "bike fuel", "flight", "parking", "toll"],
    "Shopping": ["amazon", "flipkart", "myntra", "shoes", "tshirt", "jeans", "clothes",
                 "bag", "headphones", "watch", "charger", "mobile cover", "shirt", "sandals"],
    "Bills": ["recharge", "electricity", "wifi", "internet", "rent", "mobile bill",
              "water bill", "gas cylinder", "dth", "hostel fees", "phone bill", "jio", "airtel"],
    "Education": ["books", "notebook", "pen", "xerox", "printout", "course", "udemy",
                  "coursera", "exam fees", "stationery", "project material", "lab manual",
                  "tuition", "coding course", "college fees"],
    "Entertainment": ["movie", "netflix", "spotify", "game", "pubg", "bgmi", "concert",
                      "outing", "party", "hotstar", "youtube premium", "cinema", "gaming"],
    "Health": ["medicine", "doctor", "pharmacy", "hospital", "gym", "tablet", "checkup",
               "dentist", "vitamins", "test report", "clinic", "protein"],
}

PREFIXES = ["", "", "paid for ", "bought ", "spent on ", "today "]
SUFFIXES = ["", "", " today", " yesterday", " with friends", " online", " at college"]


def _build_training_data(seed: int = 42) -> pd.DataFrame:
    """Keywords se thoda variation bana ke training data banata hai (alag CSV ki zarurat nahi)."""
    rng = random.Random(seed)
    rows = []
    for cat, words in KEYWORDS.items():
        for w in words:
            for _ in range(6):
                text = rng.choice(PREFIXES) + w + rng.choice(SUFFIXES)
                rows.append((text.strip(), cat))
    return pd.DataFrame(rows, columns=["text", "category"])


def train_category_model():
    df = _build_training_data()
    model = make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), analyzer="char_wb"),
        LogisticRegression(max_iter=1000),
    )
    model.fit(df["text"], df["category"])
    return model


def predict_category(model, text: str):
    """(category, confidence %) return karta hai."""
    probs = model.predict_proba([text])[0]
    idx = int(np.argmax(probs))
    return model.classes_[idx], round(float(probs[idx]) * 100, 1)


def predict_next_month(monthly_totals: pd.Series):
    """Pichle mahino ke total se agle mahine ka kharcha (Linear Regression).
    monthly_totals: index = month, values = total amount. Kam se kam 2 mahine chahiye."""
    if len(monthly_totals) < 2:
        return None
    X = np.arange(len(monthly_totals)).reshape(-1, 1)
    y = monthly_totals.values
    reg = LinearRegression().fit(X, y)
    return max(0.0, float(reg.predict([[len(monthly_totals)]])[0]))