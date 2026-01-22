from fastapi import FastAPI, Request
from api.schemas import (
    PredictResponse,
    AskRequest,
    AskResponse,
    RecommendResponse,
    ErrorResponse
)
from api.services import predict_service, ask_service, recommend_service
import time
from api.logger import get_logger

logger = get_logger()

app = FastAPI(
    title="Domain Intelligence System",
    version="1.0.0",
    description="Churn prediction + RAG document intelligence with citations"
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time()-start) * 1000

    logger.info(
            f"{request.method} {request.url.path} | status = {response.status_code} | {duration:.2f}ms"
            )
    return response

@app.get("/")
def root():
    return {"status": "ok", "message": "Domain Intelligence System API running"}


@app.post("/predict", response_model=PredictResponse, responses={400: {"model": ErrorResponse}})
def predict(customer: dict):
    try:
        return predict_service(customer)
    except Exception as e:
        return {"error": "Prediction failed", "details": {"message": str(e)}}


@app.post("/ask", response_model=AskResponse, responses={400: {"model": ErrorResponse}})
def ask(req: AskRequest):
    try:
        results = ask_service(req.question, top_k=req.top_k)
        return {"question": req.question, "results": results}
    except Exception as e:
        return {"error": "RAG retrieval failed", "details": {"message": str(e)}}


@app.post("/recommend", response_model=RecommendResponse, responses={400: {"model": ErrorResponse}})
def recommend(customer: dict):
    try:
        return recommend_service(customer)
    except Exception as e:
        return {"error": "Recommendation failed", "details": {"message": str(e)}}

