from fastapi import APIRouter

from .routes import bert_classifier_router, llm_classifier_router, xgboost_classifier_router

api_router = APIRouter()
api_router.include_router(bert_classifier_router)
api_router.include_router(llm_classifier_router)
api_router.include_router(xgboost_classifier_router)