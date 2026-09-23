from fastapi import FastAPI, HTTPException
from src.api.schemas import (PredictionRequest, PredictionResponse)
from src.inference.predict import predict
from src import logger


app = FastAPI(
    title="Adult Income Prediction API",
    description="Predict whether income exceeds $50K",
    version="1.0.0"
)


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict", response_model=PredictionResponse)
def make_prediction(request: PredictionRequest):
    try:
        input_data = request.model_dump(by_alias=True)
        result = predict(input_data)
        return {
            "prediction": result["predictions"][0],
            "probability": result["probabilities"][0]
        }

    except Exception as e:
        logger.exception("Prediction request failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        ) from e