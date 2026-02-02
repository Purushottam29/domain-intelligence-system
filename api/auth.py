from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader
from api.config import API_KEY

# Swagger will show this as an Authorize input
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API Key")

    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    return api_key

from fastapi import Header, HTTPException
from fastapi.security.api_key import APIKeyHeader
from api.config import API_KEY

api_key_header = APIKeyHeader(name = "X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")

    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key
