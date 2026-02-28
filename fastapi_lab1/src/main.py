from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field
from predict import predict_data

app = FastAPI(
    title="Breast Cancer Classifier API",
    description="Predicts whether a tumor is malignant or benign using a Random Forest model.",
    version="1.0.0",
)


class CancerData(BaseModel):
    """
    Pydantic model representing the 30 features from the Breast Cancer dataset.
    All measurements are computed from a digitized image of a fine needle aspirate (FNA).
    """
    mean_radius: float            = Field(..., gt=0, example=17.99)
    mean_texture: float           = Field(..., gt=0, example=10.38)
    mean_perimeter: float         = Field(..., gt=0, example=122.8)
    mean_area: float              = Field(..., gt=0, example=1001.0)
    mean_smoothness: float        = Field(..., gt=0, example=0.1184)
    mean_compactness: float       = Field(..., gt=0, example=0.2776)
    mean_concavity: float         = Field(..., gt=0, example=0.3001)
    mean_concave_points: float    = Field(..., gt=0, example=0.1471)
    mean_symmetry: float          = Field(..., gt=0, example=0.2419)
    mean_fractal_dimension: float = Field(..., gt=0, example=0.07871)

    se_radius: float              = Field(..., gt=0, example=1.095)
    se_texture: float             = Field(..., gt=0, example=0.9053)
    se_perimeter: float           = Field(..., gt=0, example=8.589)
    se_area: float                = Field(..., gt=0, example=153.4)
    se_smoothness: float          = Field(..., gt=0, example=0.006399)
    se_compactness: float         = Field(..., gt=0, example=0.04904)
    se_concavity: float           = Field(..., gt=0, example=0.05373)
    se_concave_points: float      = Field(..., gt=0, example=0.01587)
    se_symmetry: float            = Field(..., gt=0, example=0.03003)
    se_fractal_dimension: float   = Field(..., gt=0, example=0.006193)

    worst_radius: float           = Field(..., gt=0, example=25.38)
    worst_texture: float          = Field(..., gt=0, example=17.33)
    worst_perimeter: float        = Field(..., gt=0, example=184.6)
    worst_area: float             = Field(..., gt=0, example=2019.0)
    worst_smoothness: float       = Field(..., gt=0, example=0.1622)
    worst_compactness: float      = Field(..., gt=0, example=0.6656)
    worst_concavity: float        = Field(..., gt=0, example=0.7119)
    worst_concave_points: float   = Field(..., gt=0, example=0.2654)
    worst_symmetry: float         = Field(..., gt=0, example=0.4601)
    worst_fractal_dimension: float = Field(..., gt=0, example=0.1189)


class CancerResponse(BaseModel):
    prediction: str
    confidence: float


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict", response_model=CancerResponse)
async def predict_cancer(features: CancerData):
    """
    Predict whether a tumor is malignant or benign.

    Accepts all 30 Breast Cancer dataset features and returns:
    - prediction: "malignant" or "benign"
    - confidence: model's probability score for the prediction

    Raises:
        HTTPException: 500 if prediction fails.
    """
    try:
        input_data = [[
            features.mean_radius, features.mean_texture, features.mean_perimeter,
            features.mean_area, features.mean_smoothness, features.mean_compactness,
            features.mean_concavity, features.mean_concave_points, features.mean_symmetry,
            features.mean_fractal_dimension, features.se_radius, features.se_texture,
            features.se_perimeter, features.se_area, features.se_smoothness,
            features.se_compactness, features.se_concavity, features.se_concave_points,
            features.se_symmetry, features.se_fractal_dimension, features.worst_radius,
            features.worst_texture, features.worst_perimeter, features.worst_area,
            features.worst_smoothness, features.worst_compactness, features.worst_concavity,
            features.worst_concave_points, features.worst_symmetry, features.worst_fractal_dimension
        ]]

        label, confidence = predict_data(input_data)
        return CancerResponse(prediction=label, confidence=confidence)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))