import pandas as pd

# ==========================================
# CONFIGURATION: CHANGE THESE PATHS
# ==========================================
INPUT_FILE_PATH = "data/clean/cleaned_spinoff_data.csv"
OUTPUT_FILE_PATH = "data/clean/cleaned_spinoff_data.csv"


def clean_and_rename_bloomberg(input_path, output_path):
    print(f"Reading raw Bloomberg data from: {input_path}...")
    
    # 1. Load data starting from row 3 (index 3) to skip the NaNs and the old duplicate header row
    df = pd.read_csv(input_path, skiprows=3, header=None)
    
    # 2. Define your exact requested new column list
    new_columns = [
        "Action Type", 
        "Security ID", 
        "Announce/Declared Date", 
        "Effective Date", 
        "Amd Flag", 
        "Name", 
        "Spun-off Company name", 
        "Spun off Company ticker", 
        "Terms"
    ]
    
    # If Bloomberg output has trailing empty columns, slice df to match your 9 columns exactly
    df = df.iloc[:, :len(new_columns)]
    
    # Assign the requested names
    df.columns = new_columns
    
    # 3. Text parsing helper: Strip out prefixes like "Name: ", "Terms: " etc., to leave only clean data values
    columns_to_strip = {
        "Name": "Name: ",
        "Spun-off Company name": "Spun-Off Company Name: ",
        "Spun off Company ticker": "Spun-Off Company Ticker: ",
        "Terms": "Terms: "
    }
    
    for col, prefix in columns_to_strip.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(prefix, "", case=False, regex=False).str.strip()
            
    # Drop rows that are fully empty if any exist
    df.dropna(how='all', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print("\n--- Processing Complete! Cleaned Preview: ---")
    print(df.head())

    #df['Effective Date'] = pd.to_datetime(df['Effective Date'], format='%m/%d/%Y')
    #df['Announce/Declared Date'] = pd.to_datetime(df['Announce/Declared Date'], format='%m/%d/%Y')
    
    # Standard pandas syntax to sort by date (ascending)
    df = df.sort_values(by='Effective Date', ascending=True)

    # 4. Save to a fresh CSV file
    df.to_csv(output_path, index=False)
    print(f"\nSuccess! File formatted and saved to: {output_path}")

# Run the cleaning script
clean_and_rename_bloomberg(INPUT_FILE_PATH, OUTPUT_FILE_PATH)