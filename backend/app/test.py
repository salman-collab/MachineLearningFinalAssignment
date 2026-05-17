from core.preprocessing import load_data, clean_data, encode_data, normalize, train_test_split

df = load_data("data/churn.csv")
df = clean_data(df)
df = encode_data(df)
print(df.dtypes)
df = normalize(df)
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")
X = df.drop("Churn_Label", axis=1).values.tolist()
y = df["Churn_Label"].values.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y)

print("Train size:", len(X_train))
print("Test size:", len(X_test))