import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ---------- Upload Data from Frontend ---------- #
st.markdown("<h1 style='color: #336699;'>🔍 SKU Recommendation Dashboard</h1>", unsafe_allow_html=True)

st.sidebar.header("📁 Upload CSV Files")
store_file = st.sidebar.file_uploader("Upload Store Data", type=["csv"])
sku_file = st.sidebar.file_uploader("Upload SKU Data", type=["csv"])
trans_file = st.sidebar.file_uploader("Upload Transaction Data", type=["csv"])

if not (store_file and sku_file and trans_file):
    st.warning("Please upload all three data files to proceed.")
    st.stop()

store_df = pd.read_csv(store_file)
sku_df = pd.read_csv(sku_file)
trans_df = pd.read_csv(trans_file)

# ---------- Sidebar Inputs ---------- #
st.sidebar.markdown("---")
st.sidebar.header("🎯 SKU Recommendation Filters")
category = st.sidebar.selectbox("Select Store Category", store_df['category'].unique())
filtered_stores = store_df[store_df['category'] == category]['store_id'].unique()
store_id = st.sidebar.selectbox("Select Store ID", filtered_stores)
month = st.sidebar.selectbox("Select Month", range(1, 13))

# ---------- Filter Transactions for the Store and Month ---------- #
trans_df['date'] = pd.to_datetime(trans_df['date'])
trans_df['month'] = trans_df['date'].dt.month
monthly_trans = trans_df[(trans_df['store_id'] == store_id) & (trans_df['month'] == month)]

# ---------- Identify Bought SKUs ---------- #
bought_skus = monthly_trans['sku_id'].unique()
sku_df['bought'] = sku_df['sku_id'].apply(lambda x: 1 if x in bought_skus else 0)

# ---------- Feature Engineering ---------- #
sku_df['is_seasonal'] = sku_df['seasonal_tags'].apply(lambda x: 1 if str(month) in str(x) else 0)

# Merge store features
store_features = store_df[['store_id', 'area_code', 'category', 'customer_age_segment']].drop_duplicates()
selected_store_info = store_features[store_features['store_id'] == store_id]
sku_df['store_id'] = store_id
merged_df = sku_df.merge(selected_store_info, on='store_id')

# Encode categorical variables
le_area = LabelEncoder()
le_cat = LabelEncoder()
le_age = LabelEncoder()
merged_df['area_code_enc'] = le_area.fit_transform(merged_df['area_code'])
merged_df['category_enc'] = le_cat.fit_transform(merged_df['category'])
merged_df['age_seg_enc'] = le_age.fit_transform(merged_df['customer_age_segment'])

# ---------- Model Training ---------- #
features = ['price', 'profit_margin', 'is_seasonal', 'area_code_enc', 'category_enc', 'age_seg_enc']
X = merged_df[features]
y = merged_df['bought']

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X, y)

# ---------- Prediction ---------- #
merged_df['relevance_score'] = model.predict_proba(X)[:, 1]

# ---------- Quantity Recommendation ---------- #
avg_units = monthly_trans.groupby('sku_id')['quantity'].mean().reset_index()
avg_units.columns = ['sku_id', 'avg_quantity']
merged_df = merged_df.merge(avg_units, on='sku_id', how='left')
merged_df['avg_quantity'] = merged_df['avg_quantity'].fillna(0)

# Set minimum suggested quantity for unbought recommended SKUs
merged_df['suggested_qty'] = merged_df.apply(
    lambda row: max(1, int(np.ceil(row['avg_quantity'] * row['relevance_score'] * 1.2))) if row['bought'] == 0 else '-', axis=1)

# ---------- Output Results ---------- #
st.markdown(f"### 📊 Recommendations for Category: `{category}`, Store ID: `{store_id}`, Month: `{month}`")

results = merged_df[['sku_id', 'bought', 'profit_margin', 'is_seasonal', 'relevance_score', 'suggested_qty']]

# Determine top 10% threshold for recommendations
threshold = results[results['bought'] == 0]['relevance_score'].quantile(0.9)
results['Recommended'] = results.apply(lambda x: '✅' if x['bought'] == 0 and x['relevance_score'] >= threshold else '-', axis=1)
results['Purchased'] = results['bought'].apply(lambda x: '✔️' if x == 1 else '-')

# ---------- Show Debugging Info ---------- #
st.markdown("### 🛠️ Debug Info")
st.write("Min/Max Relevance Score:", results['relevance_score'].min(), results['relevance_score'].max())
st.write("SKUs Bought:", sum(results['bought']))
st.write("SKUs Recommended:", sum(results['Recommended'] == '✅'))

# ---------- Show Recommendation Tables ---------- #
st.markdown("### 📌 Top 10 Unbought Recommended SKUs")
st.dataframe(results[results['bought'] == 0].sort_values(by='relevance_score', ascending=False).head(10))

st.markdown("### 📋 Full Recommendation Table")
st.dataframe(results.sort_values(by='relevance_score', ascending=False).reset_index(drop=True))

# ---------- Export CSV ---------- #
st.download_button("⬇️ Download Recommendations as CSV", results.to_csv(index=False), file_name="recommendations.csv")

# ---------- Visualization ---------- #
st.markdown("### 📈 Top Recommended SKUs")
top_recommended = results[results['Recommended'] == '✅'].nlargest(10, 'relevance_score')

if not top_recommended.empty:
    fig, ax = plt.subplots()
    ax.barh(top_recommended['sku_id'].astype(str), top_recommended['relevance_score'], color='skyblue')
    ax.set_xlabel('Relevance Score')
    ax.set_ylabel('SKU ID')
    ax.set_title('Top 10 Recommended SKUs')
    plt.gca().invert_yaxis()
    st.pyplot(fig)
else:
    st.warning("No SKU met the recommendation criteria.")
