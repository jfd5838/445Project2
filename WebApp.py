import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Page setup
st.set_page_config(page_title="Tariff Prediction App", layout="wide")
st.title("🌍 Tariff Prediction Dashboard")

# Upload file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data", df.head())

    # Clean empty columns
    df.dropna(axis=1, how='all', inplace=True)

    # Target column
    target_column = '2022'
    if target_column not in df.columns:
        st.error("Target column '2022' not found.")
        st.stop()

    # Features
    numeric_features = df.select_dtypes(include=np.number).columns.drop(target_column, errors='ignore').tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Drop target NaNs
    df = df[df[target_column].notna()]
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Country identifier (optional)
    identifier_column = next((col for col in df.columns if 'country' in col.lower()), None)
    identifier_info = df[identifier_column] if identifier_column else pd.Series(X.index)

    # Preprocessor
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit and transform
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3),
        'Random Forest': RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'mse': mse,
            'r2': r2
        }

    # Show metrics
    st.subheader("📊 Model Evaluation")
    for name, res in results.items():
        st.markdown(f"**{name}**")
        st.write(f"MSE: {res['mse']:.4f}")
        st.write(f"R² Score: {res['r2']:.4f}")
        st.divider()

    # Plot predictions
    st.subheader("📈 Prediction Plots")

    def plot_results(y_true, y_pred, title):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        sns.scatterplot(x=y_true, y=y_pred, ax=axs[0])
        axs[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        axs[0].set_title(f"{title} - Actual vs Predicted")
        axs[0].set_xlabel("Actual")
        axs[0].set_ylabel("Predicted")

        residuals = y_true - y_pred
        sns.histplot(residuals, kde=True, ax=axs[1])
        axs[1].set_title(f"{title} - Residuals")
        axs[1].set_xlabel("Residual")

        st.pyplot(fig)

    for name, res in results.items():
        plot_results(y_test, res['y_pred'], name)
