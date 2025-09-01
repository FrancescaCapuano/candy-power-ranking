# 🍭 Candy Market Intelligence Dashboard

This project presents an interactive data-driven dashboard for analyzing candy market trends and consumer preferences. The tool is built using Python and Streamlit, and it leverages advanced data visualization and statistical modeling techniques to uncover what makes a candy popular.

The analysis is based on the [Ultimate Halloween Candy Power Ranking dataset](https://github.com/fivethirtyeight/data/tree/master/candy-power-ranking) published by FiveThirtyEight.

## 🎯 Objective

To identify the key drivers of candy popularity and support strategic decision-making for the launch of a new private-label candy product.

The dashboard answers questions like:
- What product characteristics (e.g., chocolate, caramel, fruity) are most associated with consumer preference?
- How do price and sugar levels affect win percentage?
- Which combinations of ingredients dominate high-performing products?
- How do different product clusters (e.g., chocolate bars, fruity candies) perform?
- Are there measurable brand effects beyond ingredients?

## 🧠 Key Features

### 🔍 Market Overview
- Dataset summary and top-performers
- Price categorization
- Chocolate premium effect calculation

### 🧬 Product Analysis
- Ingredient distribution breakdown
- Co-occurrence heatmaps
- Interactive visual analytics with Plotly

### 📊 Performance Insights
- KMeans clustering into product categories:
  - Chocolate Bars
  - Pralines
  - Fruit Candies
- Trend analysis of win% vs price/sugar
- Ingredient-level performance visualized by price sensitivity

### 🧾 Strategic Recommendations
- Insight cards highlighting data-backed recommendations
- Strategic segmentation by price and product type
- Ingredient pairing suggestions for market differentiation
- Mixed effects model to evaluate brand and feature-level influence

---

## 🖥️ Live Dashboard

The app is designed to run locally using Streamlit:

```bash
streamlit run app.py
