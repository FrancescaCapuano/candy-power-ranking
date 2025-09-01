# app.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.datasets import make_regression
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page settings
st.set_page_config(
    page_title="🍭 Candy Market Intelligence Dashboard",
    page_icon="🍭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .section-header {
        font-size: 2rem;
        color: #A23B72;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #F18F01;
        padding-bottom: 0.5rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# App header with emoji and styling
st.markdown(
    '<h1 class="main-header">🍭 Candy Market Intelligence Dashboard</h1>',
    unsafe_allow_html=True,
)


# Load the actual data
@st.cache_data
def load_candy_data():
    try:
        df = pd.read_csv("candy-data.csv")
        return df
    except FileNotFoundError:
        st.error("⚠️ candy-data.csv file not found. Please upload the candy dataset.")
        st.stop()


@st.cache_data
def load_brand_data():
    try:
        brands = pd.read_csv("brands.csv")
        return brands
    except FileNotFoundError:
        st.warning("⚠️ brands.csv file not found. Some brand analysis will be limited.")
        return None


# Load data
df = load_candy_data()
brands_df = load_brand_data()

# Data preprocessing (keeping your original logic)
if brands_df is not None:
    df["competitorname"] = df["competitorname"].str.strip()
    brands_df["competitorname"] = brands_df["competitorname"].str.strip()
    df = pd.merge(df, brands_df, on="competitorname", how="left")
    df_dummies = pd.get_dummies(df["brand"], prefix="brand", drop_first=False)
    df = pd.concat([df, df_dummies], axis=1)

# Binary columns identification (your original logic)
binary_cols = [
    col
    for col in df.select_dtypes(include=["int64", "float64"]).columns
    if set(df[col].dropna().unique()).issubset({0, 1})
]

# Remove non-ingredient columns from binary_cols
binary_cols = [
    col
    for col in binary_cols
    if col not in ["winpercent", "pricepercent", "sugarpercent"]
]

# Clustering (your original logic)
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(df[binary_cols])

cluster_names = {0: "Fruit Candies", 1: "Pralines", 2: "Chocolate Bars"}
df["cluster_name"] = df["cluster"].map(cluster_names)

# Sidebar with actual insights
with st.sidebar:
    st.markdown("## 📊 Executive Summary")

    # Calculate real metrics
    total_products = len(df)
    avg_win = df["winpercent"].mean()
    chocolate_avg = (
        df[df["chocolate"] == 1]["winpercent"].mean()
        if "chocolate" in df.columns
        else avg_win
    )
    chocolate_premium = chocolate_avg - avg_win

    st.markdown(
        f"""
    **Key Insights:**
    - 🍫 Chocolate premium: +{chocolate_premium:.1f}% win rate
    - 📊 {total_products} products analyzed
    - 💯 Average win rate: {avg_win:.1f}%
    - 🏆 Top performer: {df.loc[df['winpercent'].idxmax(), 'competitorname']}
    """
    )

    st.markdown("---")
    st.markdown("## 🎯 Navigation")
    section = st.selectbox(
        "Jump to Section:",
        [
            "Market Overview",
            "Product Analysis",
            "Performance Insights",
            "Strategic Recommendations",
        ],
    )

# Section 1: Market Overview
if section == "Market Overview":
    st.markdown(
        '<h2 class="section-header">🎯 Market Overview</h2>', unsafe_allow_html=True
    )

    # Dataset description (your original content)
    st.markdown(
        """
    ### Dataset Overview

    `candy-data.csv` includes attributes for 85 candies along with their rankings. For binary variables, 1 means yes, 0 means no. The data contains the following fields:

    - **chocolate**: Does it contain chocolate?
    - **fruity**: Is it fruit flavored?
    - **caramel**: Is there caramel in the candy?
    - **peanutalmondy**: Does it contain peanuts, peanut butter, or almonds?
    - **nougat**: Does it contain nougat?
    - **crispedricewafer**: Does it contain crisped rice, wafers, or a cookie component?
    - **hard**: Is it a hard candy?
    - **bar**: Is it a candy bar?
    - **pluribus**: Is it one of many candies in a bag or box?
    - **sugarpercent**: The percentile of sugar it falls under within the data set.
    - **pricepercent**: The unit price percentile compared to the rest of the set.
    - **winpercent**: The overall win percentage according to 269,000 matchups.
    """
    )

    # Create price range categories based on pricepercent
    def categorize_price(price):
        if 0 <= price < 0.25:
            return "Low"
        elif 0.25 <= price < 0.5:
            return "Medium-Low"
        elif 0.5 <= price < 0.75:
            return "Medium-High"
        else:
            return "High"

    df["price_category"] = df["pricepercent"].apply(categorize_price)

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="metric-container">
            <h3>🍭 Total Products</h3>
            <h1>{len(df)}</h1>
            <p>Analyzed</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        if brands_df is not None:
            brand_count = df["brand"].nunique()
        else:
            brand_count = "N/A"
        st.markdown(
            f"""
        <div class="metric-container">
            <h3>🏢 Brands</h3>
            <h1>{brand_count}</h1>
            <p>In Dataset</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-container">
            <h3>📈 Avg Win Rate</h3>
            <h1>{avg_win:.1f}%</h1>
            <p>Market Performance</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="metric-container">
            <h3>🍫 Chocolate Premium</h3>
            <h1>+{chocolate_premium:.1f}%</h1>
            <p>vs Market Average</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Top 10 performers (your original analysis)
    st.markdown("### 🏆 Top 10 Performers by Win Percentage")
    df_winners = df.sort_values(by="winpercent", ascending=False).head(10)

    # Enhanced table display
    display_cols = ["competitorname", "winpercent", "price_category"]
    if "brand" in df.columns:
        display_cols.append("brand")
    if "cluster_name" in df.columns:
        display_cols.append("cluster_name")

    winners_display = df_winners[display_cols].copy()
    winners_display.columns = ["Product Name", "Win %", "Price Range"] + (
        ["Brand", "Category"] if len(display_cols) > 4 else ["Brand"]
    )

    st.dataframe(
        winners_display.style.format({"Win %": "{:.1f}%"}).background_gradient(
            subset=["Win %"], cmap="RdYlGn"
        ),
        use_container_width=True,
        hide_index=True,
    )

# Section 2: Product Analysis
elif section == "Product Analysis":
    st.markdown(
        '<h2 class="section-header">🧬 Product Analysis</h2>', unsafe_allow_html=True
    )

    # Product characteristics distribution (enhanced version of your original)
    st.markdown("### Distribution of Product Characteristics")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Convert your matplotlib chart to plotly
        counts_data = []
        for col in binary_cols:
            yes_pct = (df[col].sum() / len(df)) * 100
            no_pct = 100 - yes_pct
            counts_data.append({"Characteristic": col, "Yes": yes_pct, "No": no_pct})

        counts_df = pd.DataFrame(counts_data)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Yes",
                y=counts_df["Characteristic"],
                x=counts_df["Yes"],
                orientation="h",
                marker_color="#a3c9f7",
                text=[f"{x:.1f}%" for x in counts_df["Yes"]],
                textposition="auto",
            )
        )
        fig.add_trace(
            go.Bar(
                name="No",
                y=counts_df["Characteristic"],
                x=counts_df["No"],
                orientation="h",
                marker_color="#f7b6b6",
                text=[f"{x:.1f}%" for x in counts_df["No"]],
                textposition="auto",
            )
        )

        fig.update_layout(
            barmode="stack",
            xaxis_title="Percentage",
            yaxis_title="Product Characteristic",
            template="plotly_white",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Calculate actual insights from the data
        chocolate_pct = (
            (df["chocolate"].sum() / len(df)) * 100 if "chocolate" in df.columns else 0
        )
        hard_pct = (df["hard"].sum() / len(df)) * 100 if "hard" in df.columns else 0
        pluribus_pct = (
            (df["pluribus"].sum() / len(df)) * 100 if "pluribus" in df.columns else 0
        )

        st.markdown(
            f"""
        <div class="insight-box">
            <h4>💡 Key Insights</h4>
            <ul>
                <li><strong>Chocolate</strong> appears in {chocolate_pct:.0f}% of products</li>
                <li><strong>Hard candies</strong> represent {hard_pct:.0f}% of market</li>
                <li><strong>Multi-pack products</strong> are {pluribus_pct:.0f}% of offerings</li>
                <li><strong>Premium ingredients</strong> vary significantly</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Co-occurrence analysis (your original heatmaps made interactive)
    st.markdown("### Co-occurrence Analysis")

    heatmap_option = st.radio(
        "Choose the analysis to display:",
        ("Co-occurrence Count", "Mean Win Percent for Co-occurring Pairs"),
    )

    # Your original co-occurrence matrix calculation
    co_occurrence_matrix = df[binary_cols].T.dot(df[binary_cols])

    # Your original mean winpercent matrix calculation
    mean_winpercent_matrix = pd.DataFrame(
        np.nan, index=binary_cols, columns=binary_cols
    )
    for i in binary_cols:
        for j in binary_cols:
            mask = (df[i] == 1) & (df[j] == 1)
            if mask.sum() > 0:
                mean_winpercent_matrix.loc[i, j] = df.loc[mask, "winpercent"].mean()

    mean_winpercent_matrix = mean_winpercent_matrix.round().astype("Int64")

    if heatmap_option == "Co-occurrence Count":
        fig = px.imshow(
            co_occurrence_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title="Co-occurrence Matrix",
        )
    else:
        # Create a custom heatmap with co-occurrence as color and mean winpercent as text
        fig = go.Figure(
            data=go.Heatmap(
                z=co_occurrence_matrix.values,
                x=co_occurrence_matrix.columns,
                y=co_occurrence_matrix.index,
                text=mean_winpercent_matrix.values,
                texttemplate="%{text}",
                textfont={"size": 10},
                colorscale="Blues",
                hoverongaps=False,
            )
        )
        fig.update_layout(title="Mean Win Percent for Co-occurring Pairs")

    fig.update_layout(width=800, height=500)
    st.plotly_chart(fig, use_container_width=False)

# Section 3: Performance Insights
elif section == "Performance Insights":
    st.markdown(
        '<h2 class="section-header">📊 Performance Insights</h2>',
        unsafe_allow_html=True,
    )

    # Clustering analysis (your original logic)
    st.markdown("### Candy Category Clustering")

    # Your original centroids analysis
    centroids = kmeans.cluster_centers_
    df_centroids = pd.DataFrame(
        centroids, columns=binary_cols, index=[cluster_names[i] for i in range(3)]
    )

    # Convert to percentage and round to integer
    df_centroids = df_centroids * 100
    df_centroids = df_centroids.round(0).astype(int)

    # Convert to interactive heatmap
    fig = px.imshow(
        df_centroids,
        text_auto=".0f",
        aspect="auto",
        color_continuous_scale="YlGnBu",
        title="Categories Characteristics (%)",
    )
    fig.update_layout(width=800, height=400)
    st.plotly_chart(fig, use_container_width=False)

    # Your original scatter plot made interactive
    st.markdown("### Win Percent by Price or Sugar per Candy Category")

    selected_variable = st.selectbox(
        "Select variable to analyze:",
        ["Price Percent", "Sugar Percent"],
    )

    x_variable = (
        "pricepercent" if selected_variable == "Price Percent" else "sugarpercent"
    )

    fig = px.scatter(
        df,
        x=x_variable,
        y="winpercent",
        color="cluster_name",
        title=f"Win Percent vs {selected_variable} by Candy Category",
        hover_data=["competitorname"],
        trendline="ols",
    )

    fig.update_layout(
        xaxis_title=selected_variable,
        yaxis_title="Win Percent",
        template="plotly_white",
        height=500,
        width=800,
    )

    st.plotly_chart(fig, use_container_width=False)

    # Individual ingredient analysis (your original selectbox logic)
    ingredient_cols = [
        col
        for col in df.columns
        if col
        not in [
            "competitorname",
            "winpercent",
            "pricepercent",
            "cluster",
            "cluster_name",
        ]
        and col in binary_cols
    ]

    if ingredient_cols:
        selected_ingredient = st.selectbox(
            "Select a Product Characteristic to Analyze:", ingredient_cols
        )

        df_ingredient = df[df[selected_ingredient] == 1]

        fig = px.scatter(
            df_ingredient,
            x="pricepercent",
            y="winpercent",
            hover_data=["competitorname"],
            title=f"Price vs Win Percent for Products with {selected_ingredient.title()}",
            trendline="ols",
        )

        fig.update_layout(
            xaxis_title="Price Percent",
            yaxis_title="Win Percent",
            template="plotly_white",
            height=400,
            width=700,
        )

        st.plotly_chart(fig, use_container_width=False)

# Section 4: Strategic Recommendations
elif section == "Strategic Recommendations":
    st.markdown(
        '<h2 class="section-header">🎯 Strategic Recommendations</h2>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        <div class="recommendation-box">
            <h3>🍫 Chocolate: A Key Driver of Popularity</h3>
            <ul>
                <li><strong>Chocolate-based products</strong> are the most significant driver of consumer preference</li>
                <li>Premium products like chocolate bars show strong performance</li>
                <li><strong>Price relationship</strong> is complex - careful pricing is crucial for pralines</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="recommendation-box">
            <h3>🎯 Ingredient Combinations</h3>
            <ul>
                <li>Chocolate remains the dominant factor</li>
                <li><strong>Peanutyalmondy and crispedricewafer</strong> combinations could create market niches</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="recommendation-box">
            <h3>🍓 Fruity Products: Price Sensitivity</h3>
            <ul>
                <li><strong>Fruity products</strong> tend to have lower consumer preference</li>
                <li>Show <strong>negative correlation</strong> between price and win percent</li>
                <li>Position at <strong>lower price points</strong> to maintain appeal</li>
                <li>Avoid pricing out of the market</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="recommendation-box">
            <h3>🏢 Brand Effects Matter</h3>
            <ul>
                <li><strong>Brand identity</strong> plays a role in consumer preferences</li>
                <li>The effect goes beyond product features</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Final strategic framework
    st.markdown("### 🚀 Strategic Framework")

    st.markdown(
        """
    <div class="insight-box">
        <h4>Balanced Strategy for Market Success</h4>
        <p>To succeed in the candy market, brands must balance <strong>price</strong>, <strong>ingredient composition</strong>, and <strong>consumer targeting</strong>:</p>
        <ul>
            <li><strong>Chocolate products</strong> should be positioned as premium offerings</li>
            <li><strong>Fruity products</strong> should stay in affordable price brackets</li>
            <li><strong>Ingredient combinations</strong> should be leveraged for niche market differentiation</li>
            <li><strong>Brand alignment</strong> of pricing, products, and identity builds trust and loyalty</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

# Brand summary table (if brand data is available)
if brands_df is not None and section == "Performance Insights":

    # Example data
    np.random.seed(42)

    # Fit the mixed-effects model
    model = mixedlm(
        "winpercent ~ fruity + pricepercent + chocolate + caramel + peanutyalmondy + nougat +  crispedricewafer + hard + bar + pluribus + sugarpercent",
        df,
        groups=df["brand"],
    )
    result = model.fit()

    # Extract coefficients, standard errors, t-values, and p-values
    coefficients = result.fe_params
    stderr = result.bse
    tvalues = result.tvalues
    pvalues = result.pvalues

    # Prepare a DataFrame for fixed effects
    summary_df = pd.DataFrame(
        {
            "Coefficient": coefficients,
            "Std Err": stderr,
            "t-value": tvalues,
            "p-value": pvalues,
        }
    )

    # Sort by absolute coefficient
    summary_df["abs_Coefficient"] = summary_df["Coefficient"].abs()

    # Drop the intercept and group variable ('brand') from the sorted DataFrame
    summary_df_sorted = (
        summary_df.drop(index=["Intercept", "Group Var"], errors="ignore")
        .sort_values(by="Coefficient", ascending=False)
        .drop(columns=["abs_Coefficient"])
    )

    # Format p-values for readability
    summary_df_sorted["p-value"] = summary_df_sorted["p-value"].apply(
        lambda x: f"{x:.4f}" if x < 0.0001 else f"{x:.5f}"
    )

    # Display the Fixed Effects Table
    st.markdown("### Product Characteristics Performance")

    summary_df_sorted["Product Characteristic"] = summary_df_sorted.index

    # Display the fixed effects summary table
    st.dataframe(
        summary_df_sorted[["Product Characteristic", "Coefficient"]]
        .style.format(
            {
                "Coefficient": "{:+.1f}%",  # Format for the coefficients as percentage
            }
        )
        .background_gradient(subset=["Coefficient"], cmap="RdYlGn"),
        hide_index=True,
        height=420,
        width=300,
    )

    st.markdown("### Brand Performance Summary")

    # Extract random effects for each brand
    random_effects = result.random_effects

    # Extract only the intercept value from each Series in random_effects
    intercepts = {group: effects[0] for group, effects in random_effects.items()}

    # Convert the intercepts dictionary into a DataFrame
    random_effects_df = pd.DataFrame.from_dict(
        intercepts, orient="index", columns=["Win Increment"]
    )

    # Set the index name (Optional, for readability)
    random_effects_df.index.name = "Brand"

    # Add random effect to the brand summary table
    brand_summary = (
        df.groupby("brand")
        .agg(
            total_candies=("competitorname", "count"),
            avg_winpercent=("winpercent", "mean"),
            avg_price=("pricepercent", "mean"),
        )
        .reset_index()
    )

    brand_summary["avg_winpercent_int"] = (
        brand_summary["avg_winpercent"].round().astype(int)
    )

    # Categorizing avg_price into price ranges
    def categorize_price(price):
        if price <= 0.25:
            return "Low"
        elif price <= 0.50:
            return "Medium-Low"
        elif price <= 0.75:
            return "Medium-High"
        else:
            return "High"

    # Apply price categorization
    brand_summary["avg_price_category"] = brand_summary["avg_price"].apply(
        categorize_price
    )

    brand_candy_types = df.pivot_table(
        index="brand",
        columns="cluster_name",
        values="competitorname",
        aggfunc="count",
        fill_value=0,
    )

    brand_summary = pd.merge(brand_summary, brand_candy_types, on="brand", how="left")

    brand_summary = brand_summary.rename(columns={"brand": "Brand"})
    # Merge random effects with the brand summary
    brand_summary = pd.merge(brand_summary, random_effects_df, on="Brand", how="left")

    brand_summary = brand_summary.drop(columns=["avg_winpercent", "avg_price"])

    # Format random effect as percentage increment (e.g., "+8%")
    brand_summary["Win Increment"] = brand_summary["Win Increment"].apply(
        lambda x: round(x, 1)
    )

    # Sort by win percentage and random effects
    brand_summary = brand_summary.sort_values(by="Win Increment", ascending=False)

    brand_summary.columns = [
        "Brand",
        "Total Candies",
        "Avg Win %",
        "Avg Price Category",
        "Chocolate Bars",
        "Fruit Candies",
        "Pralines",
        "Win Increment",
    ]

    # Now reorder the columns: switch "Avg Win %" and "Win Increment"
    brand_summary = brand_summary[
        [
            "Brand",
            "Total Candies",
            "Win Increment",  # Moved "Win Increment" before "Avg Win %"
            "Avg Price Category",
            "Chocolate Bars",
            "Fruit Candies",
            "Pralines",
        ]
    ]

    st.dataframe(
        brand_summary.style.format(
            {
                "Avg Win %": "{:.1f}%",
                "avg_price": "{:.1f}%",
                "Win Increment": "{:+.1f}%",
            }
        ).background_gradient(subset=["Win Increment"], cmap="RdYlGn"),
        use_container_width=True,
        hide_index=True,
    )

# Footer
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>📊 Candy Market Intelligence Dashboard | Enhanced Visual Analytics</p>
    <p>💡 Data-driven insights from actual candy preference data</p>
</div>
""",
    unsafe_allow_html=True,
)
