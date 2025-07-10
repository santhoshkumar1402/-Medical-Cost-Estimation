import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("C:/Users/SANTHOSH/OneDrive/Desktop/insurance.csv")
# Data Inspection
df.info()

# Handling Outliers - Applying Log Transformation to Charges
df['charges'] = np.log1p(df['charges'])

# Data Visualization
plt.figure(figsize=(10,5))
sns.histplot(df['charges'], bins=50, kde=True)
plt.title('Distribution of Insurance Charges')
plt.show()

sns.pairplot(df, hue='smoker')
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x='smoker', y='charges', data=df)
plt.title('Insurance Charges by Smoking Status')
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x='sex', y='charges', data=df)
plt.title('Insurance Charges by Gender')
plt.show()

plt.figure(figsize=(10,5))
sns.boxplot(x='region', y='charges', data=df)
plt.title('Insurance Charges by Region')
plt.show()

plt.figure(figsize=(10,5))
sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df)
plt.title('Insurance Charges vs BMI')
plt.show()

# Encoding categorical variables
encoder = LabelEncoder()
df['sex'] = encoder.fit_transform(df['sex'])
df['smoker'] = encoder.fit_transform(df['smoker'])
df['region'] = encoder.fit_transform(df['region'])

# Feature Selection and Train-Test Split
X = df.drop(columns=['charges'])
y = df['charges']

# Feature Scaling
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Models with Optimized Parameters
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
}

# Train & Evaluate Models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        "Mean Absolute Error": mae,
        "Mean Squared Error": mse,
        "Root Mean Squared Error": rmse,
        "R squared score": r2
    }

    print(f"\n{name} Performance:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R squared score: {r2:.4f}")

# Visualization of Model Performance
metrics = ["Mean Absolute Error", "Mean Squared Error", "Root Mean Squared Error", "R squared score"]
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for i, metric in enumerate(metrics):
    values = [results[model][metric] for model in models.keys()]
    axes[i].bar(models.keys(), values, color=['blue', 'orange', 'green'])
    axes[i].set_title(metric)
    axes[i].set_xticklabels(models.keys(), rotation=15)

plt.show()