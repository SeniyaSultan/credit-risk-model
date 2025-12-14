import pandas as pd
from src.data_processing import preprocess_data, feature_engineering

def test_preprocess_data():
    data = {'TransactionStartTime': ['2025-12-14'], 'Amount': ['100']}
    df = pd.DataFrame(data)
    df_clean = preprocess_data(df)
    assert df_clean['TransactionStartTime'].dtype == 'datetime64[ns]'
    assert df_clean['Amount'].dtype == float

def test_feature_engineering():
    data = {
        'CustomerId': ['C1', 'C1', 'C2'],
        'TransactionStartTime': ['2025-12-14', '2025-12-13', '2025-12-10'],
        'TransactionId': [1, 2, 3],
        'Amount': [100, 200, 300]
    }
    df = pd.DataFrame(data)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    rfm = feature_engineering(df)
    assert all(col in rfm.columns for col in ['Recency', 'Frequency', 'Monetary'])
