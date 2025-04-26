import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


DATA_FILE_PATH = r'C:\Users\jtdem\PycharmProjects\445Project2\TariffData.csv' # Path Changes for Each person

# Loading the Data
try:
    df = pd.read_csv(DATA_FILE_PATH)
    print(f"Successfully loaded data from: {DATA_FILE_PATH}")
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE_PATH}")
    print("Please ensure the file path is correct.")
    exit()
except Exception as e:
    print(f"An error occurred loading the CSV: {e}")
    exit()

# Cleaning
df = df.dropna(axis=1, how='all')
print("Shape after dropping empty columns:", df.shape)

# Targets / Features
target_column = '2022'
if target_column not in df.columns:
    potential_targets = [col for col in df.columns if isinstance(col, (int, str)) and '20' in str(col)]
    if potential_targets:
        target_column = potential_targets[-1]
        print(f"Warning: Target column '2022' not found. Using '{target_column}' as target.")
    else:
        raise KeyError(f"Target column '2022' not found and no likely alternative found. Available columns: {list(df.columns)}")

numeric_features = df.select_dtypes(include=np.number).columns.drop(target_column, errors='ignore').tolist()
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

if target_column in numeric_features: numeric_features.remove(target_column)
if target_column in categorical_features: categorical_features.remove(target_column)

numeric_features = [col for col in numeric_features if col in df.columns and df[col].notna().any()]
categorical_features = [col for col in categorical_features if col in df.columns and df[col].notna().any()]

# Getting the columns
country_column = None
potential_country_cols = [col for col in df.columns if 'country' in str(col).lower() or 'code' in str(col).lower() or 'entity' in str(col).lower()]
if potential_country_cols:
    cat_country_cols = [col for col in potential_country_cols if col in categorical_features]
    if cat_country_cols:
        country_column = cat_country_cols[0]
        print(f"Identified '{country_column}' as the categorical country identifier column.")
    else: # Checking other potential columns if no categorical found
        non_feature_country_cols = [col for col in potential_country_cols if col not in numeric_features and col not in categorical_features and col != target_column]
        if non_feature_country_cols:
             country_column = non_feature_country_cols[0]
             print(f"Identified '{country_column}' as a potential country identifier (not used as feature).")
        elif potential_country_cols:
             country_column = potential_country_cols[0]
             print(f"Identified '{country_column}' as potential country identifier. Checking feature status...")
             if country_column in numeric_features:
                 print(f"Warning: Treating numeric column '{country_column}' as identifier, removing from numeric features.")
                 numeric_features.remove(country_column)
             elif country_column in categorical_features:
                 print(f"Warning: Assuming categorical column '{country_column}' is identifier. Decide whether to remove from features.")


if not country_column:
    print("Warning: Could not reliably identify a country column. Using DataFrame index for identification.")

# For the pre-processing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Feature list
features_for_preprocessing = numeric_features + categorical_features
cols_to_drop = [target_column]
if country_column and country_column not in features_for_preprocessing and country_column in df.columns:
     cols_to_drop.append(country_column)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, [col for col in numeric_features if col in df.columns]),
        ('cat', categorical_transformer, [col for col in categorical_features if col in df.columns])
    ],
    remainder='drop'
    )

# Data prep / splitting
X = df.drop(columns=cols_to_drop, errors='ignore')
y = df[target_column]
valid_target_indices = y.dropna().index
X = X.loc[valid_target_indices].copy()
y = y.loc[valid_target_indices].copy()

# Store country/identifier info aligned with X
if country_column and country_column in df.columns:
    identifier_info = df.loc[valid_target_indices, country_column].copy()
elif country_column:
    identifier_info = df.loc[valid_target_indices, country_column].copy()
else:
    identifier_info = pd.Series(X.index, index=X.index, name="Index")
if X.empty or y.empty:
    raise ValueError("No data remaining after dropping NaNs in the target variable.")

# Splitting the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
identifier_info_test = identifier_info.loc[X_test.index]
identifier_col_name = identifier_info_test.name

# Fitting preprocessor and transforming our data
try:
    print("Fitting preprocessor...")
    fit_numeric_features = [col for col in numeric_features if col in X_train.columns]
    fit_categorical_features = [col for col in categorical_features if col in X_train.columns]
    # Updating preprocessor's feature lists before fitting
    preprocessor.transformers_ = [
        ('num', numeric_transformer, fit_numeric_features),
        ('cat', categorical_transformer, fit_categorical_features)
    ]
    preprocessor.fit(X_train)
    print("Transforming training data...")
    X_train_processed = preprocessor.transform(X_train)
    print("Transforming test data...")
    X_test_processed = preprocessor.transform(X_test)

except Exception as e:
    print(f"Error during preprocessing: {e}")
    print("Columns in X_train:", list(X_train.columns))
    print("Numeric features expected:", fit_numeric_features)
    print("Categorical features expected:", fit_categorical_features)
    raise e

print("Preprocessing complete.")
print("Processed training shape:", X_train_processed.shape)
print("Processed test shape:", X_test_processed.shape)

# Training and Evaluationg models
models = {}
predictions = {}
evaluations = {}

# Linear Regression Model
print("\n Training & Evaluating Linear Regression ")
lr = LinearRegression()
lr.fit(X_train_processed, y_train)
y_pred_lr = lr.predict(X_test_processed)
models['Linear Regression'] = lr
predictions['Linear Regression'] = y_pred_lr
evaluations['Linear Regression'] = {
    'MSE': mean_squared_error(y_test, y_pred_lr),
    'R2': r2_score(y_test, y_pred_lr)
}
print(f"  MSE: {evaluations['Linear Regression']['MSE']:.4f}")
print(f"  R²:  {evaluations['Linear Regression']['R2']:.4f}")

# Gradient Boosting Model
print("\n Training & Evaluating Gradient Boosting Model ")
gb = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
gb.fit(X_train_processed, y_train)
y_pred_gb = gb.predict(X_test_processed)
models['Gradient Boosting'] = gb
predictions['Gradient Boosting'] = y_pred_gb
evaluations['Gradient Boosting'] = {
    'MSE': mean_squared_error(y_test, y_pred_gb),
    'R2': r2_score(y_test, y_pred_gb)
}
print(f"  MSE: {evaluations['Gradient Boosting']['MSE']:.4f}")
print(f"  R²:  {evaluations['Gradient Boosting']['R2']:.4f}")

# Random Forest Model
print("\n Training & Evaluating Random Forest Model")
rf = RandomForestRegressor(
    n_estimators=100,   
    random_state=42,    # This is just so it is repeatable
    n_jobs=-1,
    min_samples_leaf=3  
)
rf.fit(X_train_processed, y_train)
y_pred_rf = rf.predict(X_test_processed)
models['Random Forest'] = rf
predictions['Random Forest'] = y_pred_rf
evaluations['Random Forest'] = {
    'MSE': mean_squared_error(y_test, y_pred_rf),
    'R2': r2_score(y_test, y_pred_rf)
}
print(f"  MSE: {evaluations['Random Forest']['MSE']:.4f}")
print(f"  R²:  {evaluations['Random Forest']['R2']:.4f}")


# Results
results_df = pd.DataFrame({
    identifier_col_name: identifier_info_test.reset_index(drop=True),
    'Actual': y_test.reset_index(drop=True),
    'Predicted_LR': np.round(y_pred_lr, 2),
    'Predicted_GB': np.round(y_pred_gb, 2),
    'Predicted_RF': np.round(y_pred_rf, 2)
})

print("\n Comparison of Model Predictions (First 10 Rows) ")
print(results_df.head(10))

print("\n Model Evaluation Summary ")
for name, metrics in evaluations.items():
    print(f"{name}:")
    print(f"  MSE: {metrics['MSE']:.4f}")
    print(f"  R²:  {metrics['R2']:.4f}")

print("\nScript finished.")

# Graphing visualizations for the predictions we made
print("\n Plotting Model Predictions ")
def plot_model_predictions(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 5))

    # Actual vs predicted (scatter plot)
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name} - Actual vs Predicted")

    # Actual vs predicted (Residual plot)
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True)
    plt.title(f"{model_name} - Residuals Distribution")
    plt.xlabel("Residual (Actual - Predicted)")

    plt.tight_layout()
    plt.show()

# Plotting
plot_model_predictions(y_test, y_pred_lr, "Linear Regression")
plot_model_predictions(y_test, y_pred_gb, "Gradient Boosting")
plot_model_predictions(y_test, y_pred_rf, "Random Forest")
