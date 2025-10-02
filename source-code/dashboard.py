import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings
import os

# --- Streamlit Theme Configuration ---
with open(".streamlit/config.toml", "w") as f:
    f.write("""[theme]\nbase=\"light\"\nprimaryColor=\"#2244ef\"\nbackgroundColor=\"#b9efaf\"\nsecondaryBackgroundColor=\"#e57de8\"\n""")

warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(page_title="Climate Change Dashboard", layout="wide")

# --- Custom Styling and Header Image ---
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to right, #b3e0ff 0%, #b3ffb3 100%);
    }
    [data-testid="stSidebar"] {
        background-color: #b3e0ff;
    }
    h1 {
        color: #005A9C;
        text-align: center;
    }
    h2 {
        color: #007A7A;
    }
    
    h3 {
        color: #FF4B4B;
        text-align: center;
    }
    .st-emotion-cache-1v0mbdj > img {
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    [data-testid="stTabs"] > div[role="tablist"] {
        background: linear-gradient(to right, #b3e0ff 0%, #b3ffb3 100%);
        border-radius: 8px;
        padding: 5px;
    }
    [data-testid="stTabs"] > div[role="tabpanel"] {
        background-color: #f6fcff;
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        padding: 20px;
        margin-top: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a header image
col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
with col1:
    st.image("earth_left.png", width=250)
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/97/The_Earth_seen_from_Apollo_17.jpg", width=150)
# col3 is empty for spacing
with col4:
    st.image("earth_right.png", width=250)

st.title("Climate Change Analysis Dashboard")
st.markdown("<h3 style='text-align: center; color: #FF4B4B;'>Made with ❤️ by Heart Hackers</h3>", unsafe_allow_html=True)

# --- Data Loading and Cleaning --- 
@st.cache_data
def load_and_clean_data():
    try:
        # Try to load from the original path, then from a local path
        df = pd.read_csv('climate_change_dataset[1].csv')
    except FileNotFoundError:
        st.error("Error: 'climate_change_dataset[1].csv' not found. Please make sure the dataset is in the same directory.")
        return None

    # Clean column names
    df.columns = df.columns.str.replace(r' \(.+\)', '', regex=True)
    df.rename(columns={'Extreme Weather Events': 'Extreme Events', 'Sea Level Rise': 'Sea Rise'}, inplace=True)
    
    # Convert 'Year' to numeric, handling potential errors
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df.dropna(subset=['Year'], inplace=True)
    df['Year'] = df['Year'].astype(int)

    return df

data = load_and_clean_data()

if data is not None:
    # --- Sidebar for User Input ---
    st.sidebar.header("Filters")
    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        options=data["Country"].unique(),
        default=data["Country"].unique()[:5] # Default to first 5 countries
    )

    min_year, max_year = int(data["Year"].min()), int(data["Year"].max())
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    # Filter data based on sidebar selections
    filtered_data = data[
        (data["Country"].isin(selected_countries)) &
        (data["Year"].between(year_range[0], year_range[1]))
    ]

    # --- Main Dashboard as Scrollable Sections ---
    st.header("Dataset Overview")
    st.write("Filtered data based on your selections in the sidebar.")
    st.dataframe(filtered_data)

    st.header("Climate Change Trends")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Temperature and CO2 Emissions Over Time")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='Year', y='Avg Temperature', data=filtered_data, label='Average Temperature', ax=ax)
        ax2 = ax.twinx()
        sns.lineplot(x='Year', y='CO2 Emissions', data=filtered_data, label='CO2 Emissions', color='orange', ax=ax2)
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Temperature (°C)")
        ax2.set_ylabel("CO2 Emissions (Tons/Capita)")
        fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
        st.pyplot(fig)

    with col2:
        st.subheader("Sea Level Rise and Rainfall")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='Year', y='Sea Rise', data=filtered_data, label='Sea Level Rise', ax=ax)
        ax2 = ax.twinx()
        sns.lineplot(x='Year', y='Rainfall', data=filtered_data, label='Rainfall', color='green', ax=ax2)
        ax.set_xlabel("Year")
        ax.set_ylabel("Sea Level Rise (mm)")
        ax2.set_ylabel("Rainfall (mm)")
        fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
        st.pyplot(fig)

    st.header("Correlations and Distributions")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Correlation Heatmap")
        numeric_df = filtered_data.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

    with col4:
        st.subheader("CO2 Emissions Distribution")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.histplot(filtered_data['CO2 Emissions'], bins=30, kde=True, ax=ax)
        ax.set_title('Distribution of CO2 Emissions')
        st.pyplot(fig)

    # --- Predictive Modeling ---
    st.header("Predict CO2 Emissions")
    st.write("Use the sliders to predict CO2 emissions based on other climate factors.")

    # Define features and target variable from climate_analysis2.py
    features = ['Avg Temperature', 'Sea Rise', 'Rainfall', 'Population', 'Renewable Energy', 'Extreme Events', 'Forest Area']
    target = 'CO2 Emissions'

    # Ensure all feature columns are present and numeric
    for feature in features:
        if feature not in data.columns:
            st.warning(f"Feature '{feature}' not found in the dataset. It will be ignored.")
            features.remove(feature)
        else:
            data[feature] = pd.to_numeric(data[feature], errors='coerce')
    
    data_model = data.dropna(subset=features + [target])

    X = data_model[features]
    y = data_model[target]

    if len(X) > 0:
        # Train the model
        model = LinearRegression()
        model.fit(X, y)

        # User input for prediction
        st.subheader("Input Features for Prediction")
        input_data = {}
        for feature in features:
            min_val = float(data[feature].min())
            max_val = float(data[feature].max())
            default_val = float(data[feature].mean())
            input_data[feature] = st.slider(f"Select {feature}", min_val, max_val, default_val)

        # Prediction
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)

        st.subheader("Predicted CO2 Emissions")
        st.write(f"**{prediction[0]:.2f} Tons/Capita**")
    else:
        st.warning("Not enough data to build a predictive model with the selected filters.")

else:
    st.info("Please load the data to proceed.")
