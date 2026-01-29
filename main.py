from fastapi import FastAPI
from pydantic import BaseModel, field_validator
import joblib
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInput(BaseModel):
    rooms: int
    age: int
    distance: float


class Prediction(BaseModel):
    price: float

    @field_validator("price", mode="after")
    @classmethod
    def round_to(cls, value: float) -> float:
        return round(value, 2)
    


#  Loads the model
logger.info("Loading the model...")
model = joblib.load("house_price_model.joblib")


app = FastAPI(
    title="Housing Prices Inference",
    description="Predict housing prices",
    version="1.0.0",
)


@app.post("/predict")
def predict(data: ModelInput):
    features = [[
        data.rooms,
        data.age,
        data.distance
    ]]

    prediction = model.predict(features)

    return Prediction(price=prediction[0])


@app.get("/health")
def health():
    return {"status": "OK"}