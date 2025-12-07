# _STAGE 0 : BUSINESS UNDERSTANDING_

---

## 📊 Regression Dataset: House Price Prediction

---

## 📌 Problem Statement

A real estate company is facing challenges in accurately pricing properties for sale. Currently, property valuations are conducted manually by agents using their experience and comparable sales, leading to inconsistent pricing across different agents and regions. Overpriced listings stay on the market too long, while underpriced properties result in lost revenue for both sellers and the company's commission.

The lack of a standardized, data-driven approach to property valuation has resulted in:

- Extended time-to-sale for overpriced properties
- Missed revenue opportunities from undervalued listings
- Reduced client trust due to inconsistent valuations
- Inefficient allocation of agent resources

Building on this challenge, the company aims to develop a predictive pricing model that can provide accurate, consistent property valuations based on property characteristics, location, and market conditions to optimize pricing strategies.

## 📌 Role

As a Data Scientist Team, our role involves:

- Conducting exploratory data analysis to understand property value drivers
- Identifying key features that influence house prices
- Building regression models to predict accurate property valuations
- Providing actionable insights for pricing strategy optimization

## 📌 Goals

- **Accurate Property Valuation**: Develop a model that accurately predicts house prices within an acceptable margin of error. (_MAIN_)
- **Pricing Consistency**: Ensure uniform valuation standards across all properties and regions. (_SECONDARY_)
- **Market Intelligence**: Identify key value drivers to inform renovation and investment decisions. (_SECONDARY_)

## 📌 Objectives

The ultimate goal of this project is to create a machine learning model that can:

- Predict house prices with low Mean Absolute Percentage Error (MAPE) to minimize pricing inaccuracies
- Provide price estimates with confidence intervals for negotiation guidance
- Identify the top contributing factors to property value for strategic recommendations

## 📌 Business Metrics

| Metric                                    | Description                                                     | Type        |
| ----------------------------------------- | --------------------------------------------------------------- | ----------- |
| **Mean Absolute Error (MAE)**             | Average absolute difference between predicted and actual prices | _MAIN_      |
| **Mean Absolute Percentage Error (MAPE)** | Average percentage error in predictions                         | _MAIN_      |
| **R-squared (R²)**                        | Proportion of variance explained by the model                   | _MAIN_      |
| **Time-to-Sale**                          | Average days a property stays on market before selling          | _SECONDARY_ |
| **Price-to-List Ratio**                   | Final sale price compared to initial listing price              | _SECONDARY_ |

## 📌 Dataset Overview

| Feature                  | Description                         | Type           |
| ------------------------ | ----------------------------------- | -------------- |
| `property_id`            | Unique property identifier          | ID             |
| `sqft_living`            | Living area square footage          | Numeric        |
| `sqft_lot`               | Lot size square footage             | Numeric        |
| `bedrooms`               | Number of bedrooms                  | Numeric        |
| `bathrooms`              | Number of bathrooms                 | Numeric        |
| `floors`                 | Number of floors                    | Numeric        |
| `year_built`             | Year the house was built            | Numeric        |
| `year_renovated`         | Year of last renovation (0 if none) | Numeric        |
| `condition`              | Overall condition rating (1-5)      | Ordinal        |
| `grade`                  | Construction quality rating (1-13)  | Ordinal        |
| `waterfront`             | Has waterfront view                 | Categorical    |
| `view`                   | View quality rating (0-4)           | Ordinal        |
| `city`                   | City location                       | Categorical    |
| `zipcode`                | Property zipcode                    | Categorical    |
| `latitude` / `longitude` | Geographic coordinates              | Numeric        |
| `distance_to_downtown`   | Distance to city center (miles)     | Numeric        |
| `school_rating`          | Nearby school quality (1-10)        | Numeric        |
| `crime_rate`             | Local crime rate (per 1000)         | Numeric        |
| `has_garage`             | Has garage                          | Categorical    |
| `sqft_basement`          | Basement square footage             | Numeric        |
| **`price`**              | **Target: Property sale price ($)** | **Continuous** |

## 📌 Success Criteria

- Model achieves **MAPE ≤ 15%** for price predictions
- Model achieves **R² ≥ 0.80** to explain price variability
- **MAE ≤ $50,000** for typical residential properties
- Clear identification of **top 5 price drivers** for market insights
- Reduction in average **time-to-sale by 10-15%** through optimal pricing

---

_Data generated using `data/generate_data.py` for testing and demonstration purposes._
