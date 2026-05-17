from core.preprocessing import load_data, clean_data, encode_data, convert_to_numeric, normalize, train_test_split
from algorithms.logistic_regression import LogisticRegression

df = load_data("data/churn.csv")
df = clean_data(df)
df = encode_data(df)
df = convert_to_numeric(df)
df = normalize(df)

X = df.drop("Churn Label", axis=1).values.tolist()
y = df["Churn Label"].values.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LogisticRegression(lr=0.01, epochs=500)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Predictions:", preds[:10])
print("Actual:", y_test[:10])