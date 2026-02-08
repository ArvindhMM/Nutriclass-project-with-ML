import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATA_PATH = "./data/raw/synthetic_food_dataset_imbalanced.csv"
df = pd.read_csv(DATA_PATH)

print("Initial shape:", df.shape)

df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)

numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

for col in numeric_cols:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)

print("Missing values after imputation:")
print(df.isnull().sum())

def cap_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    return df

df = cap_outliers_iqr(df, numeric_cols)


print("Outlier capping applied using IQR method.")

# Target column
target_col = "Food_Name"

# Numeric features (already identified earlier)
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

# Categorical features (excluding target)
categorical_cols = ["Meal_Type", "Preparation_Method"]

# Boolean features
boolean_cols = ["Is_Vegan", "Is_Gluten_Free"]

X = df.drop(columns=[target_col])
y = df[target_col]

numeric_transformer = Pipeline(steps=[
    ("scaler", RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
        ("bool", "passthrough", boolean_cols)
    ]
)

X_processed = preprocessor.fit_transform(X)

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

os.makedirs("./data/processed", exist_ok=True)

X_df = pd.DataFrame(X_processed)
X_df.to_csv("./data/processed/X_processed.csv", index=False)

y_df = pd.DataFrame(y_encoded, columns=["Food_Label"])
y_df.to_csv("./data/processed/y_processed.csv", index=False)

label_mapping = pd.DataFrame({
    "Food_Name": label_encoder.classes_,
    "Encoded_Label": range(len(label_encoder.classes_))
})

label_mapping.to_csv("./data/processed/label_mapping.csv", index=False)

print("Data preprocessing complete.")