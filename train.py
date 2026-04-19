# Generated from: train.ipynb
# Converted at: 2026-04-19T07:53:49.261Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd

data=pd.read_csv("retail_customer_segmentation.csv")

data.drop(['customer_id'],axis=1)

data.info()

data.isnull().sum()

data['annual_income']=data['annual_income'].fillna(data['annual_income'].median())
data['avg_monthly_spend']=data['avg_monthly_spend'].fillna(data['avg_monthly_spend'].median())
data['purchase_frequency']=data['purchase_frequency'].fillna(data['purchase_frequency'].median())
data['discount_usage_rate']=data['discount_usage_rate'].fillna(data['discount_usage_rate'].median())
data['return_rate']=data['return_rate'].fillna(data['return_rate'].median())
data['browsing_time_minutes']=data['browsing_time_minutes'].fillna(data['browsing_time_minutes'].median())
data['support_interactions']=data['support_interactions'].fillna(data['support_interactions'].median())

import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x="customer_segment",data=data)
plt.title("Customer Segment Distribution")
plt.show()


from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()
cat_features = [
    "payment_method","region","customer_segment"]

for col in cat_features:
    data[col] = le.fit_transform(data[col])

data

num_cols=["age","annual_income","months_active","avg_monthly_spend","purchase_frequency","avg_order_value","discount_usage_rate","return_rate","browsing_time_minutes","support_interactions","payment_method","region"]	

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

import pandas as pd

# Select only numerical columns (exclude target if needed)
num_cols = data.select_dtypes(include=["int64", "float64"]).columns.drop("target", errors="ignore")

# Calculate IQR
Q1 = data[num_cols].quantile(0.25)
Q3 = data[num_cols].quantile(0.75)
IQR = Q3 - Q1

# Create mask for NON-outliers
mask = ~((data[num_cols] < (Q1 - 1.5 * IQR)) |
         (data[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

# Apply mask
data_clean = data[mask]

# Results
print("Original shape:", data.shape)
print("After removing outliers:", data_clean.shape)

from scipy import stats
import numpy as np

num_cols = data.select_dtypes(include=["int64", "float64"]).columns

z_scores = np.abs(stats.zscore(data[num_cols], nan_policy='omit'))

mask = (z_scores < 3).all(axis=1)

data = data[mask]

num_cols = data.select_dtypes(include=["int64", "float64"]).columns

x = data.drop(["customer_id", "customer_segment"], axis=1)
y= data["customer_segment"]

x

y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

xgb = XGBClassifier(eval_metric='mlogloss')
xgb.fit(x_train, y_train)

y_pred_xgb = xgb.predict(x_test)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
precision = precision_score(y_test, y_pred_xgb, average='weighted')
recall = recall_score(y_test, y_pred_xgb, average='weighted')
f1 = f1_score(y_test, y_pred_xgb, average='weighted')
conf_matrix=confusion_matrix(y_test, y_pred_xgb)
print("confusion_matrix:",conf_matrix)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

import joblib
joblib.dump(xgb, "model.pkl")
joblib.dump(scaler, "scaler.pkl")  
joblib.dump(le,"label_encoder.pkl")