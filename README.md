### FASTAPI + SCIKIT inference


An example of using FastAPI with Scikit for inference. It uses the classic housing price prediction example as the backend for testing inference.


### Setup

Download dependencies:
```
uv sync --locked
```

Run `python model.py` first to train simple LinearRegression model with fake housing price dataset.

Run `uvicorn main:app` to test locally.


#### Ref

* https://machinelearningmastery.com/the-machine-learning-practitioners-guide-to-model-deployment-with-fastapi/