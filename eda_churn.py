import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv('EDA/churn_customer.csv')

# Xử lý định dạng cho các cột số có dấu phẩy
df['balance'] = df['balance'].str.replace(',', '.').astype(float)
df['estimated_salary'] = df['estimated_salary'].str.replace(',', '.').astype(float)

# 1. Kiểm tra cấu trúc dữ liệu
print("Thông tin dữ liệu:")
print(df.info())
print("\nThống kê mô tả:")
print(df.describe())
print("\nKiểm tra giá trị thiếu:")
print(df.isnull().sum())

# 2. Phân tích mô tả
# Phân phối biến mục tiêu (churn)
plt.figure(figsize=(6, 4))
sns.countplot(x='churn', data=df)
plt.title('Phân phối khách hàng rời bỏ dịch vụ')
plt.savefig('EDA/churn_distribution.png')
plt.close()

# Phân phối các biến số
numeric_cols = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']
df[numeric_cols].hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.savefig('EDA/numeric_distribution.png')
plt.close()

# Phân tích biến phân loại
categorical_cols = ['country', 'gender', 'credit_card', 'active_member']
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, hue='churn', data=df)
    plt.title(f'Phân phối {col} theo Churn')
    plt.savefig(f'EDA/{col}_by_churn.png')
    plt.close()

# 3. Phân tích chẩn đoán
# Ma trận tương quan (chỉ tính trên các cột số)
numeric_df = df[numeric_cols + ['credit_card', 'active_member', 'churn']]
plt.figure(figsize=(10, 8))
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Ma trận tương quan')
plt.savefig('EDA/correlation_matrix.png')
plt.close()

# So sánh các biến số theo churn
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='churn', y=col, data=df)
    plt.title(f'{col} theo Churn')
    plt.savefig(f'EDA/{col}_boxplot_by_churn.png')
    plt.close()