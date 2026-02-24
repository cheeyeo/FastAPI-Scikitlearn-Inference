from typing import Any
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel, field_validator
import joblib
import logging


UPLOAD_DIR = Path("uploads")

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


@app.post("/predict", response_model=Prediction)
def predict(data: ModelInput) -> Any:
    features = [[data.rooms, data.age, data.distance]]

    prediction = model.predict(features)

    return Prediction(price=prediction[0])


@app.get("/health")
def health():
    return {"status": "OK"}


# Trying out file upload
@app.post("/upload/single")
async def upload_single_file(file: UploadFile):
    if file.filename == "":
        raise HTTPException(status_code=400, detail="No file selected")

    file_path = Path(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file.size,
        "location": str(file_path),
    }


# TODO: Multiple file uploads to S3 with image resize etc using Pillow?
