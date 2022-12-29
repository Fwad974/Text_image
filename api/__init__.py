from fastapi import APIRouter

from api.main import model_api
from api.health import actuator_api

router = APIRouter()
router.include_router(router=model_api, prefix="/model")
router.include_router(router=actuator_api)

__all__ = [router]
