import yfinance as yf
import pandas as pd

df = yf.download("^GSPC", start="2000-01-01", end="2024-01-01", auto_adjust=False)

df = df.ffill()
df = df.dropna()
df = df[~df.index.duplicated()]
df = df[(df["Close"] > 0) & (df["Volume"] > 0)]
df["Returns"] = df["Close"].pct_change()
df = df.reset_index()

df['Date'] = pd.to_datetime(df['Date'])   
df['Year'] = df['Date'].dt.year           

df_12_per_year = df.groupby('Year').apply(lambda x: x.sample(n=12, random_state=1)).reset_index(drop=True)

print(df_12_per_year)
# Should see 288 rows x 9 columns printed in Terminal

df.to_csv("sp500clean_data.csv", index=False)
print("CSV saved!")