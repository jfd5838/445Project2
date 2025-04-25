import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load dataset
df = pd.read_csv('V:\My Documents\Assignments\TariffData.csv')

# Remove columns with all missing values
df = df.dropna(axis=1, how='all')

# Display basic info
print("Shape after dropping empty columns:", df.shape)
print("Columns:", list(df.columns))

# Specify the target column (update this to match one of the printed column names)
target_column = '2022'
if target_column not in df.columns:
    raise KeyError(f"Target column '{target_column}' not found. Available columns: {list(df.columns)}")

# Identify feature types
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.drop(target_column).tolist()
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Drop numeric features with all missing (if any slipped through)
numeric_features = [col for col in numeric_features if df[col].notna().any()]

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split data into features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Combine X and y to drop NaNs from target
combined = pd.concat([X, y], axis=1)
combined = combined.dropna(subset=[target_column])

# Split back into features and target
X = combined.drop(columns=[target_column])
y = combined[target_column]


# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit and transform training data, transform test data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Preprocessing complete.")
print("Processed training shape:", X_train_processed.shape)
print("Processed test shape:", X_test_processed.shape)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# === Step 1: Choose and train a model ===
# You can swap this with other models like RandomForestRegressor()
model = LinearRegression()
model.fit(X_train_processed, y_train)

# === Step 2: Make predictions ===
y_pred = model.predict(X_test_processed)

# Replace with the actual name of the column containing country info
country_column = 'Country Code'  # or 'Country', 'Country Code', etc.

# Extract the country info from X_test using the original indices
country_info = X_test[country_column].reset_index(drop=True)

# Combine predictions, actual values, and country info
results_df = pd.DataFrame({
    'Country': country_info,
    'Actual': y_test.reset_index(drop=True),
    'Predicted': np.round(y_pred, 2)
})

# Show first 10 rows
print(results_df.head(10))

# Adding a gradient boosting regressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# === Step 1: Train and evaluate Linear Regression ===
lr = LinearRegression()
lr.fit(X_train_processed, y_train)
y_pred_lr = lr.predict(X_test_processed)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr  = r2_score(y_test, y_pred_lr)
print("\nLinear Regression Evaluation:")
print(f"  MSE: {mse_lr:.2f}")
print(f"  R²:  {r2_lr:.2f}")

# === Step 2: Train and evaluate Gradient Boosting Regressor ===
gb = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train_processed, y_train)
y_pred_gb = gb.predict(X_test_processed)

# Add GB predictions to your results DataFrame
results_df['Predicted_GB'] = np.round(y_pred_gb, 2)

mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb  = r2_score(y_test, y_pred_gb)
print("\nGradient Boosting Regressor Evaluation:")
print(f"  MSE: {mse_gb:.2f}")
print(f"  R²:  {r2_gb:.2f}")

# === Optional: compare side-by-side ===
print("\nComparison:\n", results_df.head(10))