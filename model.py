import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


if __name__ == "__main__":
    data = pd.DataFrame(
        {
            "rooms": [2, 3, 4, 5, 3, 4],
            "age": [20, 15, 10, 5, 12, 7],
            "distance": [10, 8, 5, 3, 6, 4],
            "price": [100, 150, 200, 280, 180, 250],
        }
    )

    X = data[["rooms", "age", "distance"]]
    y = data["price"]

    pipeline = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])

    print("Training model...")
    pipeline.fit(X, y)

    print("Saving model after training...")
    joblib.dump(pipeline, "house_price_model.joblib")
