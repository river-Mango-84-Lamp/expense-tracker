"""Smart Expense Tracker - Streamlit app.
Run: streamlit run app.py
"""
import sqlite3
from datetime import date

import pandas as pd
import streamlit as st

from model import CATEGORIES, predict_category, predict_next_month, train_category_model

DB_PATH = "expenses.db"

st.set_page_config(page_title="Smart Expense Tracker", page_icon="💸", layout="wide")


# ---------- Database ----------
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT NOT NULL,
            exp_date TEXT NOT NULL
        )"""
    )
    return conn


def add_expense(description, amount, category, exp_date):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO expenses (description, amount, category, exp_date) VALUES (?,?,?,?)",
            (description, amount, category, str(exp_date)),
        )


def delete_expense(exp_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM expenses WHERE id = ?", (exp_id,))


def load_expenses() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT * FROM expenses ORDER BY exp_date DESC, id DESC", conn)
    if not df.empty:
        df["exp_date"] = pd.to_datetime(df["exp_date"])
    return df


# ---------- ML model (sirf ek baar train hota hai) ----------
@st.cache_resource
def get_model():
    return train_category_model()


model = get_model()

# ---------- UI ----------
st.title("💸 Smart Expense Tracker")
st.caption("Kharcha likho, category AI khud pehchan lega.")

# Sidebar: monthly budget
st.sidebar.header("⚙️ Settings")
budget = st.sidebar.number_input("Monthly budget (₹)", min_value=0, value=5000, step=500)

tab_add, tab_dash, tab_list = st.tabs(["➕ Add Expense", "📊 Dashboard", "📋 All Expenses"])

# ----- Tab 1: Add expense -----
with tab_add:
    col1, col2 = st.columns(2)
    with col1:
        desc = st.text_input("Kharcha kis cheez par hua?", placeholder="e.g. Zomato biryani")
        amount = st.number_input("Amount (₹)", min_value=1.0, value=100.0, step=10.0)
        exp_date = st.date_input("Date", value=date.today())

    with col2:
        suggested, conf = (None, 0)
        if desc.strip():
            suggested, conf = predict_category(model, desc.strip())
            st.info(f"🤖 AI ka guess: **{suggested}** ({conf}% sure)")
            if conf < 40:
                st.warning("AI thoda unsure hai, category check kar lo.")
        default_idx = CATEGORIES.index(suggested) if suggested else 0
        category = st.selectbox("Category (galat ho to badal sakte ho)", CATEGORIES, index=default_idx)

    if st.button("Save Expense", type="primary"):
        if not desc.strip():
            st.error("Pehle description likho.")
        else:
            add_expense(desc.strip(), amount, category, exp_date)
            st.success(f"Saved: {desc} - ₹{amount:.0f} ({category})")

# ----- Tab 2: Dashboard -----
df = load_expenses()
with tab_dash:
    if df.empty:
        st.info("Abhi koi expense nahi hai. Pehle kuch add karo.")
    else:
        df["month"] = df["exp_date"].dt.to_period("M").astype(str)
        this_month = date.today().strftime("%Y-%m")
        month_df = df[df["month"] == this_month]
        spent = month_df["amount"].sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Is mahine ka kharcha", f"₹{spent:,.0f}")
        m2.metric("Budget bacha", f"₹{budget - spent:,.0f}")
        m3.metric("Total entries", len(df))

        if budget > 0:
            st.progress(min(spent / budget, 1.0))
            if spent > budget:
                st.error("⚠️ Budget cross ho gaya!")
            elif spent > 0.8 * budget:
                st.warning("Budget ka 80% se zyada use ho chuka hai.")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Category-wise kharcha")
            cat_totals = df.groupby("category")["amount"].sum()
            st.bar_chart(cat_totals)
        with c2:
            st.subheader("Monthly kharcha")
            monthly = df.groupby("month")["amount"].sum().sort_index()
            st.bar_chart(monthly)

        st.subheader("🔮 Agle mahine ka andaza")
        pred = predict_next_month(monthly)
        if pred is None:
            st.info("Prediction ke liye kam se kam 2 mahine ka data chahiye.")
        else:
            st.metric("Predicted next month", f"₹{pred:,.0f}")
            st.caption("Linear Regression se, pichle mahino ke trend ke basis par.")

# ----- Tab 3: All expenses -----
with tab_list:
    if df.empty:
        st.info("Koi data nahi.")
    else:
        show = df[["id", "exp_date", "description", "category", "amount"]].copy()
        show["exp_date"] = show["exp_date"].dt.date
        st.dataframe(show, use_container_width=True, hide_index=True)

        del_id = st.number_input("Delete karne ke liye ID", min_value=0, step=1, value=0)
        if st.button("Delete"):
            if del_id in df["id"].values:
                delete_expense(int(del_id))
                st.success("Delete ho gaya. Page refresh karo.")
            else:
                st.error("Ye ID nahi mili.")

        st.download_button("Download CSV", show.to_csv(index=False), "expenses.csv", "text/csv")