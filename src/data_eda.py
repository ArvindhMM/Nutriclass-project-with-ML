import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "./data/raw/synthetic_food_dataset_imbalanced.csv"

df = pd.read_csv(DATA_PATH)
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Shape (rows, columns):")
print(df.shape)
print("\nColumn Names:")
print(df.columns.tolist())
print("\nDataset Info:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

os.makedirs("./visuals/eda", exist_ok=True)

plt.figure(figsize=(10, 5))
df['Food_Name'].value_counts().plot(kind='bar')
plt.title("Food Class Distribution")
plt.xlabel("Food Name")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("./visuals/eda/food_class_distribution.png")
plt.show()

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(f"./visuals/eda/{col}_distribution.png")
    plt.show()
