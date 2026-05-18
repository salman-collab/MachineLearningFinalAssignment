from core.preprocessing import load_data, clean_data, encode_data, normalize, train_test_split
from algorithms.linear_regression import LinearRegression

df = load_data("data/churn.csv")
df = clean_data(df)
df = encode_data(df)
df = normalize(df)
print(df.head())
# Target column
X = df.drop("Churn", axis=1).values.tolist()

# Convert target into list
y = df["Churn"].values.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = DecisionTree(max_depth=5)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Sample Predictions:", preds[:5])

print("\nDecision Tree:")
print(model.tree)
