# Import the required libraries
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# Loading data from a CSV file
file_path = '/kaggle/input/climate-change-dataset/climate_change_dataset.csv'  
data = pd.read_csv(file_path)
# Displays the first 5 rows of the dataset
data.head()
# Displays information about the dataset (number of rows, columns, data type)
data.info()
# Displays descriptive statistics of a dataset
data.describe()
# Displays the columns in the dataset
print(data.columns)
# Change column name by removing units
data.columns = data.columns.str.replace(r' \(.+\)', '', regex=True)
data.rename(columns={'Extreme Weather Events': 'Extreme Events', 'Sea Level Rise': 'Sea Rise'}, inplace=True)  # Mengubah nama kolom

# Check column names after changing
print(data.columns)
# Check if there are any missing values ​​in the entire dataset
missing_values = data.isnull().sum()

# Display the number of missing values ​​per column
print(missing_values)
# Check if there are 0 values ​​in the entire dataset
zero_values = (data == 0).sum()

# Displays the number of 0 values ​​per column
print(zero_values)
# Change the data type of a particular column if necessary
data['Year'] = pd.to_datetime(data['Year'], format='%Y').dt.year  # Change the Year column to datetime type

print(data.info())

print(data[['Year']].head())
# save the cleaned data to csv to create a dashboard later
data.to_csv('cleaned_climate_change.csv', index=False)
#import the modules needed for analysis, visualization, and calculation
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
plt.figure(figsize=(8, 6))

# Plot average temperature trend
sns.lineplot(x='Year', y='Avg Temperature', data=data, label='Average Temperature (°C)')

# Plot per capita CO2 emissions trend
sns.lineplot(x='Year', y='CO2 Emissions', data=data, label='CO2 Emissions (Tons/Capita)')

plt.title('Climate Change Trends Over the Last Decades', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend()
plt.show()
# CO2 emission distribution analysis
plt.figure(figsize=(12, 6))
sns.histplot(data=data, x='CO2 Emissions', bins=30, kde=True, color='skyblue')
plt.title('CO2 Emissions Distribution per capita', fontsize=14)
plt.xlabel('CO2 Emissions', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
# Descriptive statistics of global average temperature
country_temp_stats = data["Avg Temperature"].describe()

print(country_temp_stats)

# Get Q1, Q3, and IQR
Q1 = country_temp_stats["25%"]
Q3 = country_temp_stats["75%"]
IQR = Q3 - Q1

# Calculate lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = data[(data["Avg Temperature"] < lower_bound) | (data["Avg Temperature"] > upper_bound)]

# Count the number of outliers
outlier_count = len(outliers)

# Display the number of outliers
print(f"Number of outliers: {outlier_count}")
# Histogram plot for global temperature distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=data, x="Avg Temperature", bins=30, kde=True, color="skyblue")
plt.title("Global Mean Temperature Distribution", fontsize=16)
plt.xlabel("Average Temperature (°C)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
corr_matrix = data[['Sea Rise', 'Rainfall', 'Extreme Events']].corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()
correlation = data[['Population', 'CO2 Emissions', 'Avg Temperature', 'Rainfall', 'Sea Rise']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
correlation.to_csv('correlation_matrix.csv')
# Change to long format
correlation_long = correlation.reset_index().melt(id_vars='index', var_name='Variable 2', value_name='Correlation')
correlation_long.rename(columns={'index': 'Variable 1'}, inplace=True)

# Save to CSV file
correlation_long.to_csv('correlation_matrix_long.csv', index=False)
# Filter data for available year range (2000 to 2023)
data_recent_23_years = data[(data['Year'] >= 2000) & (data['Year'] <= 2023)]

# Aggregate number of extreme weather events per year
extreme_events_per_year = data_recent_23_years.groupby('Year')['Extreme Events'].sum().reset_index()

# Create a line plot for extreme weather event trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=extreme_events_per_year, x='Year', y='Extreme Events', marker='o', color='b')
plt.title("Extreme Weather Event Trends (2000-2023)")
plt.xlabel("Year")
plt.ylabel("Number of Extreme Weather Events")

# Highlights of the last decade (2013 - 2023)
plt.axvspan(2013, 2023, color="orange", alpha=0.3, label="Last Decade")
plt.legend()

plt.show()
# Statistical Analysis
mean_renewable = data['Renewable Energy'].mean()
std_renewable = data['Renewable Energy'].std()
var_renewable = data['Renewable Energy'].var()

print(f"Mean: {mean_renewable:.2f}")
print(f"Standard Deviation: {std_renewable:.2f}")
print(f"Variance: {var_renewable:.2f}")

# Boxplot visualization
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, y='Country', x='Renewable Energy', color='skyblue')
plt.title('Renewable Energy Variations between Countries', fontsize=14)
plt.ylabel('Country', fontsize=12)
plt.xlabel('Renewable Energy (%)', fontsize=12)
plt.show()
plt.figure(figsize=(10, 6))
data.groupby('Year')['Forest Area'].mean().plot(kind='area', color='lightgreen', alpha=0.5)
plt.title('Cumulative Forest Area Over Time')
plt.xlabel('Year')
plt.ylabel('Forest Area (%)')
plt.grid(True)
plt.show()
import statsmodels.api as sm
# Calculate the average global temperature per year
global_avg_temp = data.groupby('Year')['Avg Temperature'].mean().reset_index()

# Visualize the temperature trend
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Avg Temperature', data=global_avg_temp, marker='o')
plt.title('Global Average Temperature Trend')
plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')
plt.grid(True)
plt.show()

# Linear Regression for global temperature projection
X = global_avg_temp['Year'].values.reshape(-1, 1)  # Year as independent variable
y = global_avg_temp['Avg Temperature'].values  # Temperature as dependent variable

# Adding a constant for the intercept (linear model)
X = sm.add_constant(X)

# Building a regression model
model = sm.OLS(y, X).fit()

# Displaying the model summary
print(model.summary())

# Temperature projections for the next 10 years (current year + 10 years)
future_years = np.array([global_avg_temp['Year'].max() + i for i in range(1, 11)]).reshape(-1, 1)
future_years_with_const = sm.add_constant(future_years)
future_temp_predictions = model.predict(future_years_with_const)

# Displaying temperature projections
for year, temp in zip(future_years.flatten(), future_temp_predictions):
    print(f"Temperature projections for year {year}: {temp:.2f}°C")

# Calculates Confidence Interval (CI) for projections
predictions_with_ci = model.get_prediction(future_years_with_const)
ci_lower, ci_upper = predictions_with_ci.conf_int(alpha=0.05).T

# Visualization of temperature projections with CI
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Avg Temperature', data=global_avg_temp, marker='o', label='Data Asli')
plt.fill_between(future_years.flatten(), ci_lower, ci_upper, color='gray', alpha=0.3, label='Confidence Interval (95%)')
plt.plot(future_years, future_temp_predictions, label='Projections Temperature', color='red', linestyle='--')
plt.title('Global Mean Temperature Projection with Confidence Interval (95%)')
plt.xlabel('Year')
plt.ylabel('Mean Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()

# Model evaluation using MAE (Mean Absolute Error)
mae = np.mean(np.abs(y - model.predict(X)))
print(f"Mean Absolute Error (MAE) on training data: {mae:.2f}")

# Check model accuracy
accuracy = 100 - (mae / np.mean(y) * 100)
print(f"Model Accuracy: {accuracy:.2f}%")

# Projection accuracy of at least 85% based on MAE
if accuracy >= 85:
    print("The model has reached a minimum accuracy of 85%")
else:
    print("The model has not reached a minimum accuracy of 85%")
    