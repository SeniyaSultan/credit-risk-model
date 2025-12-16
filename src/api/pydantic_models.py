from pydantic import BaseModel

class CustomerData(BaseModel):
    total_amount: float
    avg_amount: float
    transaction_count: int
    std_amount: float
    ProductCategory: str
    ChannelId: str
    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_year: int
