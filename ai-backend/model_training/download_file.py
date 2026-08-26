import yfinance as yf

data = yf.download("GLE.PA", start="2023-01-01", end="2026-08-23")

print(data.head())
data.to_csv("GLE_stock_data.csv")
print("File saved as AAPL_stock_data.csv")   