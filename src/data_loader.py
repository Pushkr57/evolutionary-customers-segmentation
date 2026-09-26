import os
import pandas as pd
# pyrefly: ignore [missing-import, parse-error]
import 


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def find_dataset_path(data_dir: str = DATA_DIR) -> str:
    """
    Search for dataset in data_dir: checks online_retail_II.csv first, then online_retail_II.xlsx.
    Raises FileNotFoundError if neither file exists.
    """
    csv_path = os.path.join(data_dir, "online_retail_II.csv")
    xlsx_path = os.path.join(data_dir, "online_retail_II.xlsx")

    if os.path.exists(csv_path):
        return csv_path
    elif os.path.exists(xlsx_path):
        return xlsx_path
    else:
        raise FileNotFoundError(
            f"Dataset not found in '{data_dir}'. "
            f"Expected either '{csv_path}' or '{xlsx_path}'."
        )


def load_raw_data(file_path: str = None) -> pd.DataFrame:
    """
    Loads raw online retail dataset from CSV or Excel file.
    If Excel, loads all sheets and concatenates them.
    """
    if file_path is None:
        file_path = find_dataset_path()

    print(f"Loading dataset from: {file_path}")

    if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        # Load all sheets in Excel file
        excel_sheets = pd.read_excel(file_path, sheet_name=None)
        df = pd.concat(excel_sheets.values(), ignore_index=True)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format for path: {file_path}")

    # Standardize column names if needed
    column_mapping = {
        "CustomerID": "Customer ID",
        "Customer_ID": "Customer ID",
        "InvoiceNo": "Invoice",
        "UnitPrice": "Price",
    }
    df = df.rename(columns=column_mapping)

    # Ensure InvoiceDate is datetime type
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    print(f"Raw dataset shape: {df.shape}")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out:
    1. Missing Customer IDs
    2. Negative or zero Quantities
    3. Negative or zero Prices
    Calculates TotalSum per line item.
    """
    df_clean = df.copy()

    # Drop missing Customer IDs
    df_clean = df_clean.dropna(subset=["Customer ID"])

    # Cast Customer ID to int type
    df_clean["Customer ID"] = df_clean["Customer ID"].astype(int)

    # Filter out non-positive quantities and prices
    df_clean = df_clean[df_clean["Quantity"] > 0]
    df_clean = df_clean[df_clean["Price"] > 0]

    # Calculate Total Spend per line item
    df_clean["TotalSum"] = df_clean["Quantity"] * df_clean["Price"]

    print(f"Cleaned dataset shape: {df_clean.shape}")
    return df_clean


def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Recency, Frequency, Monetary (RFM) metrics and Average Order Value (AOV) per customer.

    - Recency: Days since customer's last purchase relative to max dataset date + 1 day
    - Frequency: Number of unique transactions (Invoices) per customer
    - Monetary: Total monetary spend per customer
    - AOV: Average Order Value (Monetary / Frequency)
    """
    # Reference date is 1 day after the latest date in the dataset
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("Customer ID")
        .agg(
            {
                "InvoiceDate": lambda dates: (reference_date - dates.max()).days,
                "Invoice": "nunique",
                "TotalSum": "sum",
            }
        )
        .reset_index()
    )

    rfm.columns = ["Customer ID", "Recency", "Frequency", "Monetary"]

    # Calculate Average Order Value (AOV)
    rfm["AOV"] = rfm["Monetary"] / rfm["Frequency"]

    # Round numerical metrics to 2 decimal places where appropriate
    rfm["Monetary"] = rfm["Monetary"].round(2)
    rfm["AOV"] = rfm["AOV"].round(2)

    print(f"Calculated RFM metrics for {len(rfm)} unique customers.")
    return rfm


DEFAULT_OUTPUT_PATH = os.path.join(DATA_DIR, "processed_rfm.csv")


def process_and_save_rfm(
    raw_file_path: str = None, output_file_path: str = DEFAULT_OUTPUT_PATH
) -> pd.DataFrame:
    """
    Full pipeline execution: loads raw data, cleans it, calculates RFM & AOV metrics,
    and saves the processed dataset to CSV.
    """
    df_raw = load_raw_data(raw_file_path)
    df_clean = preprocess_data(df_raw)
    rfm_df = calculate_rfm(df_clean)

    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    rfm_df.to_csv(output_file_path, index=False)
    print(f"Processed RFM dataframe successfully saved to: {output_file_path}")
    return rfm_df


if __name__ == "__main__":
    process_and_save_rfm()
