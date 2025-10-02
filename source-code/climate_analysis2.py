import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
# Load the dataset
file_path = '/kaggle/input/climate-change-dataset/climate_change_dataset.csv'
df = pd.read_csv(file_path)
df.head()
df.info()
# Check for missing values
df.isnull().sum()
# Distribution of Average Temperature
plt.figure(figsize=(10, 6))
sns.histplot(df['Avg Temperature (°C)'], bins=30, kde=True)
plt.title('Distribution of Average Temperature (°C)')
plt.xlabel('Average Temperature (°C)')
plt.ylabel('Frequency')
plt.show()
# CO2 Emissions over the Years
plt.figure(figsize=(14, 8))
sns.lineplot(data=df, x='Year', y='CO2 Emissions (Tons/Capita)', hue='Country', legend=None)
plt.title('CO2 Emissions (Tons/Capita) Over the Years')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (Tons/Capita)')
plt.show()
# Correlation Heatmap
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
# Define features and target variable
X = df[['Avg Temperature (°C)', 'Sea Level Rise (mm)', 'Rainfall (mm)', 'Population', 'Renewable Energy (%)', 'Extreme Weather Events', 'Forest Area (%)']]
y = df['CO2 Emissions (Tons/Capita)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse, r2