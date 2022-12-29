from fastapi import APIRouter
from fastapi.responses import JSONResponse

actuator_api = APIRouter()


@actuator_api.get("/health")
def health() -> JSONResponse:
    return  JSONResponse(content={"Status": "Runiing"}, status_code=222)
