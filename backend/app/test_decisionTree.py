from core.preprocessing import load_data, clean_data, encode_data, normalize, train_test_split
from algorithms.linear_regression import LinearRegression

df = load_data("data/churn.csv")
df = clean_data(df)
df = encode_data(df)
df = normalize(df)
print(df.head())
