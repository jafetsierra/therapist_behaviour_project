from fastapi import APIRouter

from .routes import bert_classifier_router

api_router = APIRouter()
api_router.include_router(bert_classifier_router)