

import pandas as pd
import numpy as np

!pip install deepchecks -q

import pandas as pd
import numpy as np
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity

df=pd.read_csv("/content/Dataset.csv")

df.head()

"""# **Data Preprocessing**"""

ds = Dataset(df, cat_features=['CASH_ADVANCE_TRX', 'PURCHASES_TRX'],

             label='TENURE')

# Run the Data Integrity Suite
integrity_suite = data_integrity()
suite_result = integrity_suite.run(ds)

# Display the results
suite_result.show()

df.isna().sum()

df.dtypes

df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(0)

df.dropna(inplace=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import pandas as pd

X = df.drop(['TENURE','CUST_ID'], axis=1)
y = df['TENURE']

# Handle categorical variables
X = pd.get_dummies(X)

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler on the training data and transform the training data
X_scaled = scaler.fit_transform(X)

# Create DataFrame with scaled features
scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Add the 'TENURE' column to the scaled DataFrame
scaled_df['TENURE'] = y.values

"""# **Exploratory Data Analysis (EDA)**"""

!pip install dabl

import dabl

df['TENURE'].unique()

eda_report = dabl.plot(df, target_col='TENURE')

# Display the EDA report
eda_report

summary = dabl.detect_types(df)
print(summary)

df.isna().sum()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Display feature importances
print(feature_importance)

"""# **Insights from Exploratory Data Analysis (EDA):**
**Target Variable Imbalance:**

The target variable is left-skewed, where the value 12 dominates the dataset with the highest number of samples.
To address this class imbalance, techniques such as:
Oversampling (e.g., SMOTE),
Undersampling (reducing the majority class),
Class weighting (adjusting model weights based on class distribution)
could be applied to mitigate this imbalance and improve model performance, especially for minority classes.

**Feature Distribution:**

Many features were observed to be right-skewed.
Applying a log transformation or other scaling techniques could help normalize the feature distribution, making the data more suitable for models that assume normality, and improving model convergence.

**PCA (Principal Component Analysis) Insights:**

From the PCA explained variance plot, it's evident that around 2-3 principal components capture a significant amount of variance in the data. The graph shows a sharp elbow at around 2-3 components, indicating that these components would be optimal for dimensionality reduction, retaining most of the important information while simplifying the dataset.

**Feature Importance:**

The top features were identified through feature importance analysis, highlighting the most influential variables for the classification task. Focusing on these features can help improve model interpretability and performance.

# **Clustering**
"""

!pip install kneed

scaled_df.isna().sum()

X=scaled_df.drop('TENURE',axis=1)
y=scaled_df['TENURE']

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from kneed import KneeLocator

# Function to plot elbow curve
def plot_elbow_curve(data, max_k):
    inertias = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.figure()
    plt.plot(range(1, max_k+1), inertias, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Curve')
    plt.show()

    kl = KneeLocator(range(1, max_k+1), inertias, curve='convex', direction='decreasing')
    return kl.elbow

# Function to perform clustering and evaluation
def perform_clustering(data, n_clusters):
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(data)
    kmeans_silhouette = silhouette_score(data, kmeans_labels)

    # Hierarchical (Agglomerative) Clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical_labels = hierarchical.fit_predict(data)
    hierarchical_silhouette = silhouette_score(data, hierarchical_labels)

    # Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(data)
    gmm_silhouette = silhouette_score(data, gmm_labels)

    return {
        'KMeans': (kmeans_labels, kmeans_silhouette),
        'Hierarchical': (hierarchical_labels, hierarchical_silhouette),
        'GMM': (gmm_labels, gmm_silhouette)
    }

# Main function that accepts X and y, performs PCA, clustering, and visualization
def clustering_analysis(X, y, max_k=10):
    # Perform dimensionality reduction using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Find optimal number of clusters using the elbow method
    optimal_k = plot_elbow_curve(X_pca, max_k)

    # Perform clustering using the optimal number of clusters
    results = perform_clustering(X_pca, optimal_k)

    # Print silhouette scores for each clustering method
    print("\nSilhouette Scores:")
    for cluster_method, (_, score) in results.items():
        print(f"  {cluster_method}: {score:.4f}")

    # Determine the best method (highest silhouette score)
    best_score = 0
    best_cluster = ''
    best_labels = None

    for cluster_method, (labels, score) in results.items():
        if score > best_score:
            best_score = score
            best_cluster = cluster_method
            best_labels = labels

    print(f"\nBest method: PCA with {best_cluster} (Silhouette Score: {best_score:.4f})")

    # Plot the best clustering result
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f"Best Clustering Result: PCA with {best_cluster}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

    # Return the optimal number of clusters, best method, and best labels
    return optimal_k, best_cluster, best_labels


optimal_k, best_cluster, best_labels = clustering_analysis(X, y, max_k=10)

"""From here we got that log transformed data is not doing good so use pca kmeans for notmal data"""

# Summarize statistics for each cluster
X['Cluster']=best_labels
cluster_summary = X.groupby('Cluster').median()
print(cluster_summary)

cluster_summary.head()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Count the number of data points in each cluster
cluster_counts = X['Cluster'].value_counts()

# Step 2: Create a pie chart
plt.figure(figsize=(8, 6))
plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
plt.title('Data Points Distribution by Cluster')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# 1. Bar Chart for Key Metrics
plt.figure(figsize=(12, 6))
key_metrics = ['BALANCE', 'PURCHASES', 'CREDIT_LIMIT', 'PAYMENTS']
cluster_summary[key_metrics].plot(kind='bar')
plt.title('Key Metrics Comparison Across Customer Segments')
plt.xlabel('Customer Segments')
plt.ylabel('Scaled Values')
plt.legend(title='Metrics')
plt.tight_layout()
plt.show()

# 2. Radar Chart for Segment Profiles
def radar_chart(df, title):
    categories = list(df.columns)
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories)

    for i, row in df.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=i)
        ax.fill(angles, values, alpha=0.1)

    plt.title(title)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.figure(figsize=(10, 10))
radar_chart(cluster_summary, 'Customer Segment Profiles')
plt.tight_layout()
plt.show()

# 4. Stacked Bar Chart for Purchase Types
purchase_types = ['ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE']
plt.figure(figsize=(10, 6))
cluster_summary[purchase_types].plot(kind='bar', stacked=True)
plt.title('Distribution of Purchase Types Across Customer Segments')
plt.xlabel('Customer Segments')
plt.ylabel('Scaled Values')
plt.legend(title='Purchase Types')
plt.tight_layout()
plt.show()

# Normalize the total usage frequency

frequency_columns = ['PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
                     'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY']

cluster_summary['TOTAL_USAGE_FREQUENCY'] = cluster_summary[frequency_columns].sum(axis=1)
max_total_frequency = cluster_summary['TOTAL_USAGE_FREQUENCY'].max()
cluster_summary['NORMALIZED_TOTAL_USAGE_FREQUENCY'] = cluster_summary['TOTAL_USAGE_FREQUENCY'] / max_total_frequency

# Create a bar plot for total usage frequency
plt.figure(figsize=(10, 6))
sns.barplot(x=cluster_summary.index, y='NORMALIZED_TOTAL_USAGE_FREQUENCY', data=cluster_summary)
plt.title('Normalized Total Usage Frequency by Customer Segment')
plt.xlabel('Customer Segments')
plt.ylabel('Normalized Total Usage Frequency')
plt.ylim(0, 1)  # Set y-axis limit from 0 to 1 for normalized values
for i, v in enumerate(cluster_summary['NORMALIZED_TOTAL_USAGE_FREQUENCY']):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

# 5. Grouped Bar Chart for Purchase Frequencies
frequency_metrics = ['PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY']
cluster_summary[frequency_metrics].plot(kind='bar', figsize=(12, 6))
plt.title('Purchase Frequencies Across Customer Segments')
plt.xlabel('Customer Segments')
plt.ylabel('Frequency (Scaled)')
plt.legend(title='Frequency Types')
plt.tight_layout()
plt.show()

"""# **Notable Insights from the graph :**

###**Segment 2 - High-Value Active Users:**

- Highest credit limit and purchase activity
- Diverse purchase behavior (one-off, installments)
- Highest payment activity
- Most frequent overall usage
- Likely the most profitable segment





###**Segment 1 - Cash Advance Focused , Low Purchase:**

- Moderate credit limit and balance
- Primarily uses cash advances
- Low overall purchase activity
- Lowest usage frequency
- May be relying on credit for cash flow needs


###Segment 0 - **Moderate Installment Users , Low Value payment:**

- Lowest credit limit and balance
- Focuses on installment purchases
- Moderate overall usage frequency
- May be using credit for specific financed purchases

# **Detailed Descriptions of Each Segment**

## **Segment 2 - High-Value Active Users**
- **Creditworthiness**: These customers have the highest credit limits, indicating a strong credit history and reliable payment behavior.
- **Usage Patterns**: They are the most active users, with frequent purchases, payments, and diverse spending habits that include a balanced mix of one-off and installment purchases.
- **Financial Stability**: Their higher balance suggests they use their credit cards regularly, likely indicating financial stability and credit-savviness.
- **Customer Base**: This segment likely includes both regular retail shoppers and business users who leverage credit for various needs.

## **Segment 1 - Cash Advance Reliant**
- **Credit and Balance**: Customers in this segment have a moderate credit limit and balance, reflecting limited financial flexibility.
- **Usage Focus**: There is an extremely high focus on cash advances, with minimal engagement in regular purchase activity. They represent the lowest overall usage frequency but the highest cash advance frequency.
- **Financial Situation**: This behavior may indicate cash flow issues or reliance on credit cards for emergency funds. They could be at higher risk of financial stress due to their reliance on cash advances.
- **Demographics**: The segment may include individuals with irregular incomes or those facing unexpected expenses.

## **Segment 0 - Cautious Installment Users**
- **Credit Characteristics**: This group has the lowest credit limit, suggesting either lower income or a shorter credit history.
- **Spending Behavior**: Their usage frequency is moderate, primarily focused on installment purchases, indicating cautious spending habits.
- **Balance and Purchases**: They maintain a low overall balance with minimal one-off purchases, reflecting a strategy to use credit for specific financed purchases rather than general consumption.
- **Target Audience**: This segment likely includes younger customers or those working on building their credit. They may also be price-sensitive individuals looking for ways to manage larger purchases.

# **Proposed Marketing Strategies for Each Segment**

## **Segment 2 - High-Value Active Users**
- **Tiered Rewards Program**: Implement a rewards program that offers cashback or points across various purchase categories, incentivizing diverse spending.
- **Premium Card Options**: Introduce premium credit card options with exclusive benefits, such as airport lounge access and concierge services, to cater to their lifestyle.
- **Targeted Promotions**: Offer tailored promotions for high-value purchases or travel-related spending, aligning with their usage patterns.
- **Personalized Loyalty Programs**: Develop a loyalty program with personalized offers based on their diverse spending habits, enhancing customer retention.
- **Financial Management Tools**: Provide tools to help users optimize credit usage, track rewards, and manage payments effectively, promoting responsible usage.

## **Segment 1 - Cash Advance Reliant**
- **Financial Education Program**: Launch initiatives focusing on budget management and alternatives to cash advances, educating customers on better financial practices.
- **Low-Interest Personal Loans**: Introduce specialized low-interest loan products as an alternative to cash advances, providing a more sustainable financing option.
- **Early Warning System**: Implement systems to identify customers at risk of falling into debt cycles, offering proactive assistance and support.
- **Financial Counseling Partnerships**: Collaborate with financial counseling services to offer free consultations, aiding customers in managing their finances.
- **Balance Transfer Promotions**: Create promotions encouraging balance transfers to consolidate high-interest cash advance debt into manageable payments.
- **Mobile App Features**: Develop an app feature that allows users to easily track cash advance usage and associated fees, fostering awareness and responsible borrowing.

## **Segment 0 - Cautious Installment Users**
- **Low-Interest Installment Plans**: Promote low-interest installment plans for larger purchases, encouraging more frequent use of their credit cards.
- **Credit Limit Increases**: Offer incremental credit limit increases tied to positive payment behavior, building trust and encouraging usage.
- **Credit-Builder Program**: Implement programs that provide educational content and gradually introduce benefits as customers build their credit.
- **Cash-Back for Installments**: Create a cash-back program specifically rewarding installment purchases, motivating cautious users to engage more with their cards.
- **Exclusive Retail Partnerships**: Partner with popular retailers to offer exclusive installment deals, appealing to price-sensitive customers.
- **Purchase Planning Features**: Develop app functionalities that help users plan and manage their installment purchases effectively, enhancing their financial management.
"""
