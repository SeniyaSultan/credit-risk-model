import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path: str) -> pd.DataFrame:
    """Load CSV/JSON data into a DataFrame."""
    df = pd.read_csv(path)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning and type conversions."""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['Amount'] = df['Amount'].astype(float)
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Example: RFM feature creation."""
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (df['TransactionStartTime'].max() - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    })
    scaler = StandardScaler()
    rfm[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(rfm)
    return rfm
