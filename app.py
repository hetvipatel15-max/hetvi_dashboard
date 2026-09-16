import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="📊", layout="wide")

st.title("📊 Customer Segmentation Dashboard")
st.caption("Interactive analysis of the iFood customer dataset")

# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("ifood_df.csv")

df = load_data().copy()

# -----------------------------
# Create useful labels
# -----------------------------
def get_marital(row):
    for col, label in [
        ("marital_Married", "Married"),
        ("marital_Single", "Single"),
        ("marital_Together", "Together"),
        ("marital_Divorced", "Divorced"),
        ("marital_Widow", "Widow"),
    ]:
        if col in row and row[col] == 1:
            return label
    return "Unknown"

def get_education(row):
    for col, label in [
        ("education_Basic", "Basic"),
        ("education_2n Cycle", "2n Cycle"),
        ("education_Graduation", "Graduation"),
        ("education_Master", "Master"),
        ("education_PhD", "PhD"),
    ]:
        if col in row and row[col] == 1:
            return label
    return "Unknown"

df["Marital Status"] = df.apply(get_marital, axis=1)
df["Education"] = df.apply(get_education, axis=1)

# Age groups
df["Age Group"] = pd.cut(
    df["Age"],
    bins=[0, 30, 40, 50, 60, 200],
    labels=["≤30", "31–40", "41–50", "51–60", "60+"]
)

# Total purchases
purchase_cols = [
    "NumWebPurchases", "NumCatalogPurchases",
    "NumStorePurchases", "NumDealsPurchases"
]
df["Total Purchases"] = df[purchase_cols].sum(axis=1)

# -----------------------------
# Customer segmentation
# -----------------------------
cluster_features = [
    "Income", "Recency", "MntTotal",
    "NumWebPurchases", "NumCatalogPurchases",
    "NumStorePurchases", "NumDealsPurchases"
]

X = df[cluster_features].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fixed 4 clusters: simple and presentation-friendly
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Give clusters meaningful names based on spending
cluster_summary = df.groupby("Cluster").agg(
    Avg_Income=("Income", "mean"),
    Avg_Spending=("MntTotal", "mean"),
    Avg_Recency=("Recency", "mean"),
    Customers=("Cluster", "size")
).reset_index()

# Rank by spending and assign labels
cluster_summary = cluster_summary.sort_values("Avg_Spending", ascending=False)
names = ["High Value", "Regular", "Low Value", "Occasional"]
cluster_name_map = {
    cluster: names[i] for i, cluster in enumerate(cluster_summary["Cluster"])
}
df["Segment"] = df["Cluster"].map(cluster_name_map)

# Give every customer a simple, human-friendly ID for searching/reference
df["Customer ID"] = "CUST-" + (df.index + 1).astype(str).str.zfill(5)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("🔎 Filters")

segments = st.sidebar.multiselect(
    "Customer Segment",
    sorted(df["Segment"].dropna().unique()),
    default=sorted(df["Segment"].dropna().unique())
)

education = st.sidebar.multiselect(
    "Education",
    sorted(df["Education"].unique()),
    default=sorted(df["Education"].unique())
)

marital = st.sidebar.multiselect(
    "Marital Status",
    sorted(df["Marital Status"].unique()),
    default=sorted(df["Marital Status"].unique())
)

age_groups = st.sidebar.multiselect(
    "Age Group",
    list(df["Age Group"].cat.categories),
    default=list(df["Age Group"].cat.categories)
)

filtered = df[
    df["Segment"].isin(segments)
    & df["Education"].isin(education)
    & df["Marital Status"].isin(marital)
    & df["Age Group"].isin(age_groups)
].copy()

if filtered.empty:
    st.warning("No customers match the selected filters. Please change the filters.")
    st.stop()

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_search = st.tabs(["📊 Overview", "🔍 Customer Search"])

# ===================================================================
# TAB 1: OVERVIEW (existing dashboard content)
# ===================================================================
with tab_overview:

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Customers", f"{len(filtered):,}")
    c2.metric("💰 Avg Income", f"${filtered['Income'].mean():,.0f}")
    c3.metric("🛒 Total Spending", f"${filtered['MntTotal'].sum():,.0f}")
    c4.metric("🛍️ Avg Purchases", f"{filtered['Total Purchases'].mean():.1f}")

    st.divider()

    # Charts row 1
    left, right = st.columns(2)

    with left:
        segment_counts = (
            filtered["Segment"]
            .value_counts()
            .rename_axis("Segment")
            .reset_index(name="Customers")
        )
        fig = px.pie(
            segment_counts,
            names="Segment",
            values="Customers",
            hole=0.45,
            title="Customer Segments"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        product_cols = {
            "Wines": "MntWines",
            "Fruits": "MntFruits",
            "Meat": "MntMeatProducts",
            "Fish": "MntFishProducts",
            "Sweets": "MntSweetProducts",
            "Gold": "MntGoldProds",
        }
        spending = pd.DataFrame({
            "Product": list(product_cols.keys()),
            "Spending": [filtered[c].sum() for c in product_cols.values()]
        }).sort_values("Spending", ascending=False)

        fig = px.bar(
            spending,
            x="Product",
            y="Spending",
            title="Spending by Product Category",
            text_auto=".2s"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Charts row 2
    left, right = st.columns(2)

    with left:
        channel = pd.DataFrame({
            "Channel": ["Web", "Catalog", "Store", "Deals"],
            "Purchases": [
                filtered["NumWebPurchases"].sum(),
                filtered["NumCatalogPurchases"].sum(),
                filtered["NumStorePurchases"].sum(),
                filtered["NumDealsPurchases"].sum(),
            ]
        })

        fig = px.bar(
            channel,
            x="Channel",
            y="Purchases",
            title="Purchases by Channel",
            text_auto=".2s"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        age_data = (
            filtered.groupby("Age Group", observed=False)
            .agg(Spending=("MntTotal", "mean"), Customers=("Age", "size"))
            .reset_index()
        )

        fig = px.bar(
            age_data,
            x="Age Group",
            y="Spending",
            title="Average Spending by Age Group",
            text_auto=".0f"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Segment comparison
    st.subheader("📌 Segment Comparison")

    summary = (
        filtered.groupby("Segment")
        .agg(
            Customers=("Segment", "size"),
            Avg_Income=("Income", "mean"),
            Avg_Spending=("MntTotal", "mean"),
            Avg_Recency=("Recency", "mean"),
            Avg_Purchases=("Total Purchases", "mean"),
        )
        .reset_index()
    )

    summary.columns = [
        "Segment", "Customers", "Avg Income",
        "Avg Spending", "Avg Recency", "Avg Purchases"
    ]

    st.dataframe(
        summary.style.format({
            "Avg Income": "${:,.0f}",
            "Avg Spending": "${:,.0f}",
            "Avg Recency": "{:.1f}",
            "Avg Purchases": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True
    )

    # Interactive scatter
    st.subheader("🔍 Customer Behaviour Explorer")

    x_axis = st.selectbox(
        "X-axis",
        ["Income", "Age", "Recency", "MntTotal", "Total Purchases"],
        index=0
    )

    y_axis = st.selectbox(
        "Y-axis",
        ["MntTotal", "Income", "Age", "Recency", "Total Purchases"],
        index=0
    )

    fig = px.scatter(
        filtered,
        x=x_axis,
        y=y_axis,
        color="Segment",
        hover_data=["Age", "Income", "MntTotal", "Total Purchases"],
        title=f"{y_axis} vs {x_axis}"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Download filtered data
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered Customer Data",
        data=csv,
        file_name="filtered_customer_data.csv",
        mime="text/csv"
    )

    st.caption("Customer segmentation is generated using K-Means clustering on income, recency, spending and purchase behaviour.")

# ===================================================================
# TAB 2: CUSTOMER SEARCH
# ===================================================================
with tab_search:

    st.subheader("Search Customers")

    # --- Free-text search across the searchable text-like fields ---
    search_text = st.text_input(
        "Search customer ID, segment, education or marital status",
        placeholder="Example: High Value, PhD, Married, CUST-00042..."
    )

    # --- Tag-style multiselect filters (mirrors the "Sentiment" tag filter) ---
    tag_col1, tag_col2 = st.columns(2)

    with tag_col1:
        search_segments = st.multiselect(
            "Segment",
            sorted(filtered["Segment"].dropna().unique()),
            default=sorted(filtered["Segment"].dropna().unique()),
            key="search_segments"
        )

    with tag_col2:
        search_education = st.multiselect(
            "Education",
            sorted(filtered["Education"].dropna().unique()),
            default=sorted(filtered["Education"].dropna().unique()),
            key="search_education"
        )

    # --- Numeric range filters ---
    range_col1, range_col2, range_col3 = st.columns(3)

    with range_col1:
        income_min, income_max = float(filtered["Income"].min()), float(filtered["Income"].max())
        income_range = st.slider(
            "Income range ($)",
            min_value=income_min,
            max_value=income_max,
            value=(income_min, income_max)
        )

    with range_col2:
        spend_min, spend_max = float(filtered["MntTotal"].min()), float(filtered["MntTotal"].max())
        spend_range = st.slider(
            "Total spending range ($)",
            min_value=spend_min,
            max_value=spend_max,
            value=(spend_min, spend_max)
        )

    with range_col3:
        age_min, age_max = int(filtered["Age"].min()), int(filtered["Age"].max())
        age_range = st.slider(
            "Age range",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max)
        )

    # --- Apply all search/filter criteria ---
    search_results = filtered[
        filtered["Segment"].isin(search_segments)
        & filtered["Education"].isin(search_education)
        & filtered["Income"].between(income_range[0], income_range[1])
        & filtered["MntTotal"].between(spend_range[0], spend_range[1])
        & filtered["Age"].between(age_range[0], age_range[1])
    ].copy()

    if search_text.strip():
        text_cols = ["Customer ID", "Segment", "Education", "Marital Status", "Age Group"]
        combined = search_results[text_cols].astype(str).agg(" ".join, axis=1).str.lower()
        search_results = search_results[combined.str.contains(search_text.strip().lower())]

    st.markdown(f"**Showing {len(search_results):,} matching customers.**")

    display_cols = [
        "Customer ID", "Segment", "Education", "Marital Status", "Age Group",
        "Age", "Income", "MntTotal", "Total Purchases", "Recency"
    ]

    st.dataframe(
        search_results[display_cols].rename(columns={
            "MntTotal": "Total Spending",
            "Total Purchases": "Purchases",
            "Recency": "Days Since Last Purchase"
        }).style.format({
            "Income": "${:,.0f}",
            "Total Spending": "${:,.0f}",
        }),
        use_container_width=True,
        hide_index=True
    )

    search_csv = search_results[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Search Results",
        data=search_csv,
        file_name="customer_search_results.csv",
        mime="text/csv"
    )
