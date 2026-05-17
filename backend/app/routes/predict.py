from fastapi import APIRouter
from app.core.preprocessing import load_data, clean_data, encode_data, convert_to_numeric, normalize, train_test_split
from app.algorithms.linear_regression import LinearRegression
from app.algorithms.logistic_regression import LogisticRegression
from app.core.metrics import accuracy, confusion_matrix, precision, recall

router = APIRouter()

@router.get("/models")
def get_models():
    return {
        "models": ["linear_regression", "logistic_regression"]
    }


@router.post("/predict")
def predict(model_name: str):
    
    # Load and preprocess data
    df = load_data("app/data/churn.csv")
    df = clean_data(df)
    df = encode_data(df)
    df = convert_to_numeric(df)
    df = normalize(df)

    # Split
    X = df.drop("Churn Label", axis=1).values.tolist()
    y = df["Churn Label"].values.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Model selection
    if model_name == "linear_regression":
        model = LinearRegression(lr=0.01, epochs=200)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        return {
            "model": model_name,
            "predictions": preds[:10]
        }

    elif model_name == "logistic_regression":
        model = LogisticRegression(lr=0.01, epochs=200)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        return {
            "model": model_name,
            "accuracy": accuracy(y_test, preds),
            "confusion_matrix": confusion_matrix(y_test, preds),
            "precision": precision(y_test, preds),
            "recall": recall(y_test, preds),
            "sample_predictions": preds[:10]
        }

    else:
        return {"error": "Invalid model"}