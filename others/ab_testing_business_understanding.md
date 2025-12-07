# _STAGE 0 : BUSINESS UNDERSTANDING_

---

## 📊 A/B Testing Dataset: Website Conversion Experiment

---

## 📌 Problem Statement

An e-commerce company is looking to improve its website conversion rate to increase revenue. The current website design has been in place for over a year, and the marketing team believes that changes to the landing page could significantly improve user engagement and purchase behavior. However, making changes without proper testing could negatively impact the user experience and result in lost revenue.

The company needs a systematic approach to:

- Test new website designs without risking overall performance
- Make data-driven decisions about which design performs better
- Measure the true impact of changes with statistical confidence
- Avoid false conclusions from random fluctuations in user behavior

Building on this challenge, the company has implemented an A/B test where users are randomly assigned to either the control group (existing design) or treatment group (new design) to measure the impact on conversion rates.

## 📌 Role

As a Data Scientist Team, our role involves:

- Designing and validating the A/B test methodology
- Conducting statistical analysis to determine experiment results
- Calculating sample sizes and experiment duration requirements
- Providing actionable recommendations based on statistical evidence

## 📌 Goals

- **Data-Driven Decision Making**: Determine whether the new design significantly improves conversion rates. (_MAIN_)
- **Statistical Rigor**: Ensure conclusions are based on proper statistical methods with controlled error rates. (_MAIN_)
- **Business Impact Quantification**: Estimate the revenue impact of implementing the winning variant. (_SECONDARY_)

## 📌 Objectives

The ultimate goal of this project is to conduct a rigorous A/B test analysis that can:

- Determine if there is a statistically significant difference between control and treatment groups
- Calculate the effect size and confidence intervals for the conversion rate difference
- Provide clear go/no-go recommendations for implementing the new design
- Segment analysis to understand if effects vary across user groups

## 📌 Business Metrics

| Metric                                 | Description                                  | Type        |
| -------------------------------------- | -------------------------------------------- | ----------- |
| **Conversion Rate (%)**                | Percentage of users who completed a purchase | _MAIN_      |
| **Statistical Significance (p-value)** | Probability result occurred by chance        | _MAIN_      |
| **Effect Size (Lift %)**               | Relative improvement in conversion rate      | _MAIN_      |
| **Revenue per User**                   | Average revenue generated per visitor        | _SECONDARY_ |
| **Time on Page**                       | User engagement indicator (seconds)          | _SECONDARY_ |
| **Pages Viewed**                       | Depth of user engagement                     | _SECONDARY_ |

## 📌 Dataset Overview

| Feature          | Description                            | Type        |
| ---------------- | -------------------------------------- | ----------- |
| `user_id`        | Unique user identifier                 | ID          |
| `group`          | Experiment group (control/treatment)   | Categorical |
| `converted`      | Did user convert? (0/1)                | Binary      |
| `revenue`        | Revenue from user (0 if not converted) | Numeric     |
| `time_on_page`   | Time spent on landing page (seconds)   | Numeric     |
| `pages_viewed`   | Number of pages viewed in session      | Numeric     |
| `device`         | User device type                       | Categorical |
| `browser`        | User browser                           | Categorical |
| `country`        | User country                           | Categorical |
| `age_group`      | User age bracket                       | Categorical |
| `returning_user` | Is returning visitor?                  | Boolean     |
| `timestamp`      | Time of visit                          | Datetime    |

## 📌 Success Criteria

- Achieve **statistical power ≥ 80%** to detect a 5% relative lift
- Use **significance level α = 0.05** for hypothesis testing
- Calculate **95% confidence intervals** for conversion rate difference
- Minimum detectable effect (MDE) of **2-3 percentage points**
- Clear recommendation with **p-value < 0.05** for implementation decision

## 📌 Hypothesis

- **Null Hypothesis (H₀)**: There is no difference in conversion rates between control and treatment groups
- **Alternative Hypothesis (H₁)**: The treatment group has a higher conversion rate than the control group

---

_Data generated using `data/generate_data.py` for testing and demonstration purposes._
