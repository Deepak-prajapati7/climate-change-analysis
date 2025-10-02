import pandas as pd

def load_climate_change_dataset(file_path):
    """Load the climate change dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None

def main():
    # Specify the path to the CSV file
    file_path = 'climate_change_dataset.csv'  # Update this path if needed
    
    # Load the dataset
    climate_data = load_climate_change_dataset(file_path)
    
    if climate_data is not None:
        # Display the first few rows of the dataset
        print(climate_data.head())
        
        # Display summary statistics
        print(climate_data.describe())
        
        # Count unique countries in the dataset
        unique_countries = climate_data['Country'].nunique()
        print(f"Number of unique countries: {unique_countries}")

if __name__ == "__main__":
    main()