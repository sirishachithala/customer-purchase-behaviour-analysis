
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# MODULE 1: DATA COLLECTION AND LOADING
# --------------------------------------------------

FILE_NAME = "customer_purchase_data.csv"
OUTPUT_DIR = "analysis_outputs"
GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")

os.makedirs(GRAPH_DIR, exist_ok=True)

if not os.path.exists(FILE_NAME):
    np.random.seed(42)

    categories = [
        "Electronics",
        "Clothing",
        "Beauty & Health",
        "Home Decor",
        "Books"
    ]

    n = 1000

    df_new = pd.DataFrame({
        "Customer ID": np.random.randint(1001, 1051, n),
        "Category": np.random.choice(categories, n),
        "Purchase Amount": np.round(
            np.random.uniform(500, 50000, n), 2
        ),
        "Quantity": np.random.randint(1, 6, n),
        "Date": pd.to_datetime(
            np.random.randint(
                pd.Timestamp("2026-01-01").value // 10**9,
                pd.Timestamp("2026-09-19").value // 10**9,
                n
            ),
            unit="s"
        )
    })

    df_new.to_csv(FILE_NAME, index=False)
    print("Sample dataset created successfully.")

df = pd.read_csv(FILE_NAME)

print("\n========== MODULE 1: DATA LOADING ==========")
print("\nFirst Five Records:")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

print("\nOriginal Data Types:")
print(df.dtypes)


# --------------------------------------------------
# MODULE 2: DATA CLEANING
# --------------------------------------------------

print("\n========== MODULE 2: DATA CLEANING ==========")

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows Before Cleaning:", df.duplicated().sum())

rows_before = len(df)

df = df.drop_duplicates().copy()

df = df.dropna(
    subset=[
        "Customer ID",
        "Category",
        "Purchase Amount",
        "Quantity",
        "Date"
    ]
).copy()

rows_after = len(df)

print("\nRows Before Cleaning:", rows_before)
print("Rows After Cleaning:", rows_after)
print("Rows Removed:", rows_before - rows_after)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows After Cleaning:", df.duplicated().sum())


# --------------------------------------------------
# MODULE 3: DATA PREPROCESSING
# --------------------------------------------------

print("\n========== MODULE 3: DATA PREPROCESSING ==========")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Purchase Amount"] = pd.to_numeric(
    df["Purchase Amount"], errors="coerce"
)
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["Customer ID"] = pd.to_numeric(
    df["Customer ID"], errors="coerce"
)

df = df.dropna(
    subset=[
        "Customer ID",
        "Category",
        "Purchase Amount",
        "Quantity",
        "Date"
    ]
).copy()

df["Customer ID"] = df["Customer ID"].astype(int)
df["Month"] = df["Date"].dt.to_period("M").astype(str)

print("\nData Types After Preprocessing:")
print(df.dtypes)

print("\nEarliest Transaction Date:", df["Date"].min())
print("Latest Transaction Date:", df["Date"].max())

print("\nPurchase Amount Descriptive Statistics:")
print(df["Purchase Amount"].describe())

print("\nCleaned Dataset Shape:", df.shape)

df.to_csv(
    os.path.join(OUTPUT_DIR, "cleaned_customer_purchase_data.csv"),
    index=False
)


# --------------------------------------------------
# MODULE 4: CUSTOMER BEHAVIOUR ANALYSIS
# --------------------------------------------------

print("\n========== MODULE 4: CUSTOMER BEHAVIOUR ==========")

customer_analysis = df.groupby("Customer ID").agg(
    Total_Spending=("Purchase Amount", "sum"),
    Average_Spending=("Purchase Amount", "mean"),
    Transaction_Count=("Purchase Amount", "count"),
    Total_Quantity=("Quantity", "sum")
).reset_index()

customer_analysis = customer_analysis.sort_values(
    "Total_Spending", ascending=False
)

print("\nTop Five Customers:")
print(customer_analysis.head(5).round(2).to_string(index=False))

top_customer = customer_analysis.iloc[0]

print("\nHighest-Spending Customer ID:", top_customer["Customer ID"])
print("Their Total Spending: ₹{:,.2f}".format(
    top_customer["Total_Spending"]
))

customer_analysis.to_csv(
    os.path.join(OUTPUT_DIR, "customer_analysis.csv"),
    index=False
)


# --------------------------------------------------
# MODULE 5: PRODUCT CATEGORY ANALYSIS
# --------------------------------------------------

print("\n========== MODULE 5: CATEGORY ANALYSIS ==========")

category_analysis = df.groupby("Category").agg(
    Total_Revenue=("Purchase Amount", "sum"),
    Average_Purchase=("Purchase Amount", "mean"),
    Total_Quantity=("Quantity", "sum"),
    Transactions=("Purchase Amount", "count")
).reset_index()

total_revenue = df["Purchase Amount"].sum()

category_analysis["Revenue_Share_Percent"] = (
    category_analysis["Total_Revenue"] / total_revenue * 100
)

category_analysis = category_analysis.sort_values(
    "Total_Revenue", ascending=False
)

print("\nCategory-wise Revenue and Sales Share:")
print(category_analysis.round(2).to_string(index=False))

leading_category = category_analysis.iloc[0]

print("\nLeading Category:", leading_category["Category"])
print("Category Revenue: ₹{:,.2f}".format(
    leading_category["Total_Revenue"]
))
print("Revenue Share: {:.2f}%".format(
    leading_category["Revenue_Share_Percent"]
))

category_analysis.to_csv(
    os.path.join(OUTPUT_DIR, "category_analysis.csv"),
    index=False
)


# --------------------------------------------------
# MONTHLY REVENUE ANALYSIS
# --------------------------------------------------

monthly_analysis = df.groupby("Month").agg(
    Monthly_Revenue=("Purchase Amount", "sum"),
    Transactions=("Purchase Amount", "count")
).reset_index()

monthly_analysis = monthly_analysis.sort_values("Month")

print("\n========== MONTHLY REVENUE ANALYSIS ==========")
print(monthly_analysis.round(2).to_string(index=False))

monthly_analysis.to_csv(
    os.path.join(OUTPUT_DIR, "monthly_analysis.csv"),
    index=False
)


# --------------------------------------------------
# MODULE 6: DATA VISUALIZATION
# --------------------------------------------------

print("\n========== MODULE 6: DATA VISUALIZATION ==========")

sns.set_theme(style="whitegrid")


def save_graph(filename):
    plt.tight_layout()
    plt.savefig(
        os.path.join(GRAPH_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# 1. Category-wise Revenue
plt.figure(figsize=(9, 5))
sns.barplot(
    data=category_analysis,
    x="Category",
    y="Total_Revenue"
)
plt.title("Category-wise Revenue")
plt.xlabel("Product Category")
plt.ylabel("Total Revenue")
plt.xticks(rotation=25)
save_graph("01_category_revenue.png")


# 2. Category-wise Purchase Distribution (Pie Chart)
plt.figure(figsize=(8, 8))
plt.pie(
    category_analysis["Transactions"],
    labels=category_analysis["Category"],
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Category-wise Purchase Distribution")
save_graph("02_category_distribution.png")


# 3. Purchase Amount Distribution
plt.figure(figsize=(9, 5))
sns.histplot(
    data=df,
    x="Purchase Amount",
    bins=30,
    kde=True
)
plt.title("Purchase Amount Distribution")
plt.xlabel("Purchase Amount")
plt.ylabel("Frequency")
save_graph("03_purchase_distribution.png")


# 4. Purchase Amount by Category (Box Plot)
plt.figure(figsize=(9, 5))
sns.boxplot(
    data=df,
    x="Category",
    y="Purchase Amount"
)
plt.title("Purchase Amount by Category")
plt.xticks(rotation=25)
save_graph("04_category_boxplot.png")


# 5. Monthly Revenue Trend
plt.figure(figsize=(10, 5))
sns.lineplot(
    data=monthly_analysis,
    x="Month",
    y="Monthly_Revenue",
    marker="o"
)
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
save_graph("05_monthly_revenue.png")


# 6. Transactions vs Total Spending per Customer
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=customer_analysis,
    x="Transaction_Count",
    y="Total_Spending"
)
plt.title("Customer Transactions vs Total Spending")
plt.xlabel("Number of Transactions")
plt.ylabel("Total Spending")
save_graph("06_customer_transactions_vs_spending.png")


# 7. Top Five Spending Customers
top_five = customer_analysis.head(5)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=top_five,
    x="Customer ID",
    y="Total_Spending"
)
plt.title("Top Five Customers by Total Spending")
plt.xlabel("Customer ID")
plt.ylabel("Total Spending")
save_graph("07_top_customers.png")


# 8. Average Purchase by Category
plt.figure(figsize=(9, 5))
sns.barplot(
    data=category_analysis,
    x="Category",
    y="Average_Purchase"
)
plt.title("Average Purchase Amount by Category")
plt.xticks(rotation=25)
save_graph("08_average_category_purchase.png")


# 9. Monthly Transaction Count
plt.figure(figsize=(10, 5))
sns.barplot(
    data=monthly_analysis,
    x="Month",
    y="Transactions"
)
plt.title("Monthly Transaction Count")
plt.xlabel("Month")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)
save_graph("09_monthly_transactions.png")


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n========== PROJECT SUMMARY ==========")
print("Total Transactions:", len(df))
print("Number of Customers:", df["Customer ID"].nunique())
print("Number of Categories:", df["Category"].nunique())
print("Total Revenue: ₹{:,.2f}".format(total_revenue))
print("Average Purchase: ₹{:,.2f}".format(
    df["Purchase Amount"].mean()
))
print("Median Purchase: ₹{:,.2f}".format(
    df["Purchase Amount"].median()
))
print("Average Quantity: {:.2f}".format(df["Quantity"].mean()))

print("\nInterpretation:")
print(
    f"1. Highest-spending customer: "
    f"{int(top_customer['Customer ID'])}."
)
print(
    f"2. Leading revenue category: {leading_category['Category']}."
)
print(
    f"3. Average transaction value: "
    f"₹{df['Purchase Amount'].mean():,.2f}."
)

print("\nAnalysis completed successfully!")
print("CSV reports saved in:", OUTPUT_DIR)
print("Graphs saved in:", GRAPH_DIR)
