from pydantic import BaseModel
from typing import List, Dict, Any

class PredictResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk: str


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AskResult(BaseModel):
    text: str
    source: str
    page: int
    distance: float

class AskResponse(BaseModel):
    question: str
    results: List[AskResult]

class RetentionAction(BaseModel):
    title: str
    details: str
    eligibility: str

class RecommendResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk: str
    message: str
    recommended_text: str
    sources: List[str]
    actions: List[RetentionAction]

class ErrorResponse(BaseModel):
    error: str
    details: Dict[str, Any] | None = None

