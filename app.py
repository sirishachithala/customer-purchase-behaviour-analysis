
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(
    page_title="Customer Purchase Behaviour Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Purchase Behaviour Analysis")
st.write("Explore customer purchases, category revenue and monthly sales.")

# Load dataset
import sqlite3

db_path = Path("customer_purchases.db")

if not db_path.exists():
    st.error("Database not found! Run python init_db.py first.")
    st.stop()

with sqlite3.connect(db_path) as conn:
    df = pd.read_sql_query("SELECT * FROM purchases", conn)

# Clean and preprocess
required_columns = [
    "Customer ID", "Category", "Purchase Amount", "Quantity", "Date"
]

if not all(col in df.columns for col in required_columns):
    st.error("Dataset columns do not match the expected format.")
    st.stop()

df = df.drop_duplicates().dropna(subset=required_columns).copy()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Purchase Amount"] = pd.to_numeric(
    df["Purchase Amount"], errors="coerce"
)
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df = df.dropna(subset=required_columns)

# Sidebar filters
st.sidebar.header("Filter Data")

categories = sorted(df["Category"].unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Select categories",
    categories,
    default=categories
)

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

filtered = df[df["Category"].isin(selected_categories)].copy()

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered = filtered[
        (filtered["Date"].dt.date >= start_date)
        & (filtered["Date"].dt.date <= end_date)
    ]

if filtered.empty:
    st.warning("No transactions match the selected filters.")
    st.stop()

# Summary metrics
total_revenue = filtered["Purchase Amount"].sum()
total_transactions = len(filtered)
total_customers = filtered["Customer ID"].nunique()
average_purchase = filtered["Purchase Amount"].mean()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Revenue", f"₹{total_revenue / 100000:.2f}L")
c2.metric("Transactions", f"{total_transactions:,}")
c3.metric("Customers", f"{total_customers:,}")
c4.metric("Average Purchase", f"₹{average_purchase:,.0f}")

# Category analysis
st.subheader("🛍️ Category Analysis")

category_data = filtered.groupby("Category").agg(
    Revenue=("Purchase Amount", "sum"),
    Transactions=("Purchase Amount", "count")
).reset_index().sort_values("Revenue", ascending=False)

left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=category_data,
        x="Category",
        y="Revenue",
        ax=ax
    )
    ax.set_title("Revenue by Category")
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)
    plt.close(fig)

with right:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(
        category_data["Transactions"],
        labels=category_data["Category"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Purchase Distribution")
    st.pyplot(fig)
    plt.close(fig)

# Customer analysis
st.subheader("👥 Top 5 Customers")

customer_data = filtered.groupby("Customer ID").agg(
    Total_Spending=("Purchase Amount", "sum"),
    Average_Spending=("Purchase Amount", "mean"),
    Transaction_Count=("Purchase Amount", "count")
).reset_index()

top_customers = customer_data.sort_values(
    "Total_Spending", ascending=False
).head(5)

st.dataframe(top_customers, width="stretch")

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(
    data=top_customers,
    x="Customer ID",
    y="Total_Spending",
    ax=ax
)
ax.set_title("Top 5 Customers by Spending")
st.pyplot(fig)
plt.close(fig)

# Monthly trend
st.subheader("📈 Monthly Revenue Trend")

filtered["Month"] = filtered["Date"].dt.to_period("M").astype(str)

monthly_data = filtered.groupby("Month").agg(
    Revenue=("Purchase Amount", "sum")
).reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(
    data=monthly_data,
    x="Month",
    y="Revenue",
    marker="o",
    ax=ax
)
ax.set_title("Monthly Revenue")
ax.tick_params(axis="x", rotation=45)
st.pyplot(fig)
plt.close(fig)

# Transaction table and download
st.subheader("📋 Transaction Data")
st.dataframe(filtered, use_container_width=True)

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Filtered Data (CSV)",
    data=csv_data,
    file_name="filtered_customer_purchases.csv",
    mime="text/csv"
)

st.success("Dashboard loaded successfully!")
