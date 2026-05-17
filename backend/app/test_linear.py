from core.preprocessing import load_data, clean_data, encode_data, normalize, train_test_split
from algorithms.linear_regression import LinearRegression

df = load_data("data/churn.csv")
df = clean_data(df)
df = encode_data(df)
df = normalize(df)
print(df.head())
# Regression target
X = df.drop("Monthly Charge", axis=1).values.tolist()
y = df["Monthly Charge"].values.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression(lr=0.01, epochs=500)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Sample Predictions:", preds[:5])