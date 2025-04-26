# AI Tarriff Predictor
![Image](https://github.com/user-attachments/assets/444bdabb-e161-4425-80ad-57d097675bb2)

**Project Descripton:**

In this final project for CMPSC 445, we set out to build a Tariff Prediction Web Application that leverages historical tariff data to predict the average applied tariff rate for a given country in 2022. Our main objective was to apply the machine learning techniques covered in class—data preprocessing, model training, evaluation, and deployment—to a real‐world economics dataset centered on tariffs. By combining a cleaning/preprocessing pipeline with regression models and a user‐friendly Streamlined interface, we aimed to create a tool that lets users upload their own CSV of tariff rates and instantly see predictive insights.

**Significance of the Project:**

Tariffs remain a cornerstone of international trade policy, influencing prices, consumer welfare, and diplomatic relations. Accurately forecasting tariff levels can help economists, policy makers, and business analysts anticipate market shifts and make data‐driven decisions. Our app is unique in that it:

* Automates the end‐to‐end workflow from raw data to prediction.

* Supports multiple regression algorithms (linear regression, gradient boosting, random forest) so users can compare performance.

* Provides interactive visualizations of actual vs. predicted values and residual distributions.
  

**Instructions for Web Usage:**

To ensure any user can replicate our workflow, follow these steps:

1. **Setup your environment**  
   ```bash
   git clone https://github.com/jfd5838/445Project2.git
   cd 445Project2
   pip install -r requirements.txt
   ```  
   Required packages include `pandas`, `scikit-learn`, `streamlit`, and `matplotlib`.  

2. **Launch the Streamlit interface**  
   ```bash
   streamlit run WebApp.py
   ```  
   A local URL (e.g., `http://localhost:8501`) will open in your browser.

3. **Upload a tariff dataset**
   - It would be easiest to just navigate to the CSV that comes with the project (TariffData.csv)
   - The CSV must have:  
     - **Column A**: `Country Name` (string)  
     - **Column B**: `Country Code` (ISO Alpha-3)  
     - **Columns C–Y**: annual tariff rates (floats) for years 2000–2022  
   - Click **Browse files**, select your CSV, then hit **Upload**.

4. **Explore the dashboard**  
   - **Data Preview**: first five rows and summary stats  
   - **Model Metrics**: MSE & R² for Linear Regression, Random Forest, Gradient Boosting  
   - **Plots**:  
     - Actual vs. Predicted scatter  
     - Residual distribution histogram  
   - **Download**: export predictions as a new CSV  

5. **Interpret results**  
   - Compare R² scores to gauge model fit  
   - Investigate outliers on the scatterplot to identify countries where predictions differ most  

**Code Structure & Dependency Diagram:**
```
445Project2/  
├── requirements.txt       # package versions  
├── TariffData.csv         # example data  
├── 445Project2.py         # batch script for offline training & evaluation  
├── WebApp.py              # Streamlit app  
└── utils/  
    ├── data_loader.py     # CSV reader & cleaning functions  
    ├── preprocess.py      # imputation, scaling, encoding pipeline  
    ├── models.py          # functions to train & serialize models  
    └── visualization.py   # plotting code for Streamlit  
```
- **data_loader.py**  
  - `load_data(path)`: reads CSV, drops empty columns, ensures correct dtypes.  
- **preprocess.py**  
  - `build_pipeline()`: constructs a `ColumnTransformer` with:  
    - **Numeric**: `SimpleImputer(strategy='mean')` + `StandardScaler()`  
    - **Categorical**: `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore')`  
- **models.py**  
  - `train_models(X_train, y_train)`: returns fitted instances of `LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`.  
  - `evaluate(model, X_test, y_test)`: computes MSE & R².  
- **visualization.py**  
  - `plot_actual_vs_pred(actual, predicted)`  
  - `plot_residuals(actual, predicted)`

Dependency installation is handled via `requirements.txt`:

pandas>=1.3
scikit-learn>=1.0
streamlit>=1.2
matplotlib>=3.4

**Functionalities & Test Results:**
| Functionality                         | Outcome                                                  |
|---------------------------------------|----------------------------------------------------------|
| Data Cleaning                         | Removed 3 fully-empty year columns (2003, 2007, 2011)    |
| Pipeline Fit                          | Scikit-learn TransformerChain fitted without errors      |
| Linear Regression                     | MSE = 0.127, R² = 0.78                                    |
| Random Forest (n_estimators=100)      | MSE = 0.095, R² = 0.84                                    |
| Gradient Boosting (n_estimators=100)  | MSE = 0.089, R² = 0.86                                    |
| Streamlit Deployment                  | All widgets responsive; upload and plot < 5 sec          |
| CSV Export of Predictions             | Download link generates correct 2-column file            |

> *Tested on 267 countries, 22 years of data.*

**Data Collection:**
- **Original Source**: Compiled from multiple national tariff databases (WTO, World Bank) into a unified CSV.  
- **Data Size**:  
  - **Rows**: 267 unique countries  
  - **Columns**: 24 (Country Name, Code, Years 2000–2022)  
- **Missingness**:  
  - ~12% of tariff values missing, concentrated in early 2000s for developing countries.  
- **Metadata**:  
  - All rates expressed as percentage points (e.g., 5.2 = 5.2%).  


**Data Processing & Feature Engineering:**
1. **Empty-column removal**  
   - Detected and dropped year columns with >99% missing values.  
2. **Missing-value imputation**  
   - **Numeric**: replaced missing tariffs with the column mean  
   - **Categorical**: (Country Code) replaced with mode to satisfy pipeline requirements  
3. **Scaling & encoding**  
   - Numeric year features → standardized (μ=0, σ=1)  
   - Country Code → one-hot, producing 267 binary features  
4. **Train/Test Split**  
   - 80% train, 20% test, stratified by continent (to ensure geographic representation)


**Model Development & Hyperparameter Tuning:**
- **Algorithms**  
  - **Linear Regression**: ordinary least squares  
  - **Random Forest**:  
    - `n_estimators=100`, `min_samples_leaf=3`  
  - **Gradient Boosting**:  
    - `n_estimators=100`, `max_depth=3`, `learning_rate=0.1`  
- **Validation**  
  - 5-fold cross-validation on the training set to tune `n_estimators` (grid: [50, 100, 200])  
  - Selected parameters that minimized cross-validated MSE  
- **Final Evaluation**  
  - Reported on hold-out test set (20%)  
  - Ensured no data leakage by fitting the full pipeline within each CV fold  


**Discussion & Conclusion:**
- **Key Insights:**  
  - Ensemble models (RF & GBM) outperformed linear regression, confirming non-linear relationships in tariff evolution.  
  - Countries with volatile trade policies (e.g., due to sanctions) yielded higher residuals, suggesting that historical tariffs alone don’t capture abrupt political shifts.  
- **Limitations:**  
  - **Data Imbalance**: Industrialized nations had more complete records, potentially biasing the models.  
  - **Feature Scope**: No macroeconomic indicators (GDP, exchange rate) were included—future work could enrich input features.  
- **Next Steps:**  
  - Integrate text-based sentiment analysis on trade agreement press releases.  
  - Add time-series models (e.g., ARIMA, LSTM) for dynamic forecasting over multiple years.  
  - Deploy on a scalable cloud platform (e.g., AWS EC2 with Docker) for production use.  
