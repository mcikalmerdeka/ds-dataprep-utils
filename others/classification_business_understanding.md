# _STAGE 0 : BUSINESS UNDERSTANDING_

---

## 📊 Classification Dataset: Customer Churn Prediction

---

## 📌 Problem Statement

A telecommunications company is experiencing significant customer attrition (churn), where existing customers discontinue their services and switch to competitors. The current reactive approach—addressing churn only after customers have already left—has proven costly and ineffective. Customer acquisition costs are typically 5-7x higher than retention costs, making each churned customer a substantial financial loss.

The company's customer service team relies on intuition and basic rules to identify at-risk customers, resulting in missed opportunities to intervene before customers leave. Without a data-driven approach, the company cannot effectively prioritize retention efforts or allocate resources efficiently.

Building on this challenge, the company aims to develop a predictive system that can identify customers at high risk of churning before they leave, enabling proactive retention strategies based on historical customer behavior and service usage patterns.

## 📌 Role

As a Data Scientist Team, our role involves:

- Conducting exploratory data analysis to understand customer behavior patterns
- Identifying key factors that contribute to customer churn
- Building predictive models to flag at-risk customers
- Providing actionable insights for business decision-making

## 📌 Goals

- **Proactive Churn Prevention**: Identify at-risk customers before they churn to enable timely intervention. (_MAIN_)
- **Customer Lifetime Value Optimization**: Focus retention efforts on high-value customers most likely to churn. (_SECONDARY_)
- **Resource Allocation Efficiency**: Prioritize marketing and retention budgets toward customers who need it most. (_SECONDARY_)

## 📌 Objectives

The ultimate goal of this project is to create a machine learning model that can:

- Predict customer churn with high recall to minimize missed at-risk customers (false negatives are costly)
- Provide probability scores for churn risk to enable tiered intervention strategies
- Identify the top contributing factors to churn for targeted retention campaigns

## 📌 Business Metrics

| Metric                            | Description                                               | Type        |
| --------------------------------- | --------------------------------------------------------- | ----------- |
| **Churn Rate (%)**                | Percentage of customers who churned in a given period     | _MAIN_      |
| **Customer Retention Rate (%)**   | Percentage of customers retained after intervention       | _MAIN_      |
| **Customer Lifetime Value (CLV)** | Projected revenue from a customer over their relationship | _SECONDARY_ |
| **Cost per Acquisition (CPA)**    | Cost to acquire a new customer vs. retain existing        | _SECONDARY_ |

## 📌 Dataset Overview

| Feature                 | Description                           | Type        |
| ----------------------- | ------------------------------------- | ----------- |
| `customer_id`           | Unique customer identifier            | ID          |
| `age`                   | Customer age                          | Numeric     |
| `gender`                | Customer gender                       | Categorical |
| `tenure_months`         | Months as a customer                  | Numeric     |
| `monthly_charges`       | Monthly billing amount                | Numeric     |
| `total_charges`         | Cumulative charges                    | Numeric     |
| `contract_type`         | Service contract type                 | Categorical |
| `payment_method`        | Payment method used                   | Categorical |
| `support_tickets`       | Number of support requests            | Numeric     |
| `account_balance`       | Current account balance               | Numeric     |
| `internet_service`      | Type of internet service              | Categorical |
| `online_security`       | Has online security add-on            | Categorical |
| `satisfaction_score`    | Customer satisfaction (1-5)           | Numeric     |
| `days_since_last_login` | Engagement metric                     | Numeric     |
| `num_products`          | Number of products subscribed         | Numeric     |
| `referral_source`       | How customer was acquired             | Categorical |
| **`churn`**             | **Target: Did customer churn? (0/1)** | **Binary**  |

## 📌 Success Criteria

- Model achieves **Recall ≥ 80%** for churn class (minimize false negatives)
- Model achieves **Precision ≥ 60%** to avoid excessive false alarms
- Reduction in monthly churn rate by **15-20%** through proactive interventions
- Clear identification of **top 5 churn indicators** for actionable insights

---

_Data generated using `data/generate_data.py` for testing and demonstration purposes._
