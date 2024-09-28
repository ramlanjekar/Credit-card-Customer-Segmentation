# Customer Segmentation Analysis

## Project Overview

This project focuses on customer segmentation using clustering techniques and exploratory data analysis (EDA) to identify distinct customer groups and derive actionable insights. The dataset consists of various features related to credit card usage, including balance, purchases, credit limits, payments, and more.

### Objectives

- **Preprocess the data** by handling missing values and scaling features.
- **Perform Exploratory Data Analysis (EDA)** to understand feature distributions, class imbalances, and key characteristics.
- **Implement clustering algorithms** such as KMeans, Hierarchical Clustering, and Gaussian Mixture Models (GMM) to identify customer segments.
- **Visualize and interpret the clusters** to derive insights about different customer groups.
- **Propose marketing strategies** for each customer segment based on the identified patterns.

## Steps and Methods

### 1. Data Preprocessing
- **Handling Missing Values**: Filled missing values in the `MINIMUM_PAYMENTS` column and removed remaining NaN values.
- **Feature Scaling**: Applied MinMaxScaler to normalize the data, ensuring that all features are on a similar scale.
- **Categorical Variables**: Handled categorical variables by converting them into dummy variables.

### 2. Exploratory Data Analysis (EDA)
- **Target Variable Imbalance**: The `TENURE` variable is left-skewed, with a dominant class (value 12). Addressing this imbalance can improve model performance for minority classes.
- **Feature Distribution**: Many features are right-skewed. Transformation techniques like log transformation could help normalize the data.
- **PCA Insights**: PCA showed that 2-3 principal components capture a significant amount of variance in the data, indicating these components are optimal for dimensionality reduction.

### 3. Clustering Analysis
- **KMeans Clustering**: Used the elbow method to find the optimal number of clusters.
- **Hierarchical Clustering**: Performed Agglomerative Clustering to identify clusters.
- **Gaussian Mixture Models (GMM)**: Applied GMM to further explore customer segments.
- **Silhouette Scores**: Evaluated each clustering method using silhouette scores to determine the best performing clustering technique.
- **Best Method**: PCA with KMeans clustering was identified as the best method based on silhouette scores.

### 4. Visualization of Clusters
- **Scatter Plots**: Visualized clusters using scatter plots of PCA components.
- **Pie Charts**: Showed the distribution of data points across different clusters.
- **Bar Charts**: Compared key metrics like balance, purchases, credit limits, and payments across customer segments.
- **Radar Charts**: Illustrated customer segment profiles based on different financial behaviors.
- **Stacked Bar Charts**: Displayed the distribution of purchase types (e.g., one-off purchases, installments, cash advances) across segments.

## Insights from the Clusters

### Segment 2 - High-Value Active Users
- **Characteristics**: High credit limits, diverse spending behavior, frequent payments.
- **Marketing Strategy**: Introduce premium credit card options, tiered rewards programs, and personalized loyalty programs to encourage engagement.

### Segment 1 - Cash Advance Reliant
- **Characteristics**: Focus on cash advances, minimal purchase activity, moderate credit limits.
- **Marketing Strategy**: Offer financial education programs, low-interest personal loans, and balance transfer promotions.

### Segment 0 - Cautious Installment Users
- **Characteristics**: Low credit limits, focused on installment purchases, moderate usage frequency.
- **Marketing Strategy**: Promote low-interest installment plans, offer credit-building programs, and partner with retailers for exclusive installment deals.

## Proposed Marketing Strategies for Each Segment

### Segment 2 - High-Value Active Users
- **Tiered Rewards Program**: Offer rewards for high-value purchases.
- **Premium Card Options**: Introduce premium credit card options with exclusive benefits.
- **Personalized Loyalty Programs**: Tailor loyalty programs to fit customer behavior.
- **Financial Management Tools**: Provide tools for effective credit management.

### Segment 1 - Cash Advance Reliant
- **Financial Education Program**: Educate customers on managing their finances better.
- **Low-Interest Personal Loans**: Provide alternatives to cash advances.
- **Balance Transfer Promotions**: Help customers manage high-interest debts.
- **Early Warning System**: Identify at-risk customers and offer assistance.

### Segment 0 - Cautious Installment Users
- **Low-Interest Installment Plans**: Encourage credit card use with low-interest plans.
- **Credit Limit Increases**: Reward responsible behavior with gradual credit limit increases.
- **Exclusive Retail Partnerships**: Partner with retailers to offer installment deals.

## Conclusion

This project successfully identified customer segments through clustering and derived actionable insights that can guide marketing strategies. By focusing on key metrics and customer behaviors, businesses can better tailor their offerings to meet the needs of different customer groups.

## Dependencies

- **Python** (version 3.x)
- **pandas**
- **NumPy**
- **scikit-learn**
- **deepchecks**
- **dabl**
- **kneed**
- **matplotlib**
- **seaborn**

To install dependencies, run:

```bash
pip install pandas numpy scikit-learn deepchecks dabl kneed matplotlib seaborn
