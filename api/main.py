from fastapi import FastAPI, Request, Depends
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
from api.auth import verify_api_key
from fastapi.security.api_key import APIKeyHeader

logger = get_logger()

app = FastAPI(
    title="Domain Intelligence System",
    version="1.0.0",
    description="Churn prediction + RAG document intelligence with citations",
    openapi_tags=[
        {
            "name": "secured",
            "description": "Endpoints protected by API Key"
        }
    ]
)
api_key_scheme = APIKeyHeader(name="X-API-Key")

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
def predict(customer: dict, _:str = Depends(verify_api_key)):

    try:
        return predict_service(customer)
    except Exception as e:
        return {"error": "Prediction failed", "details": {"message": str(e)}}

@app.post("/ask", response_model=AskResponse, responses={400: {"model": ErrorResponse}})
def ask(req: AskRequest, _:str = Depends(verify_api_key)):
    try:
        results = ask_service(req.question, top_k=req.top_k)
        return {"question": req.question, "results": results}
    except Exception as e:
        return {"error": "RAG retrieval failed", "details": {"message": str(e)}}

@app.post("/recommend", response_model=RecommendResponse, responses={400: {"model": ErrorResponse}})
def recommend(customer: dict, _:str = Depends(verify_api_key)):
    try:
        return recommend_service(customer)
    except Exception as e:
        return {"error": "Recommendation failed", "details": {"message": str(e)}}

