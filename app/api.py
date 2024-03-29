import logging
import torch
from fastapi import FastAPI
from transformers import DistilBertTokenizer
from contextlib import asynccontextmanager

from config import ENV_VARIABLES, MODELS_DIR
from pipeline.bert.utils import DistillBERTClass
from pipeline.llm.classifier import LlmClassifier
from app.endpoints import api_router
from pipeline.llm import load_yaml_dict, load_txt

__VERSION__ = "0.1.0"
STAGE = ENV_VARIABLES['STAGE']

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model_for_inference = DistillBERTClass()
        model_for_inference.load_state_dict(torch.load(MODELS_DIR / 'distilbert_finetuned.pth'))
        model_for_inference.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_for_inference.to(device)
        logging.info("BERT model loaded successfully")

        llm_classifier = LlmClassifier(
            chain_config=load_yaml_dict(ENV_VARIABLES["LLM_CONFIG_PATH"]),
            context=load_txt(ENV_VARIABLES["CONTEXT_PATH"]),
            classes=load_txt(ENV_VARIABLES["CLASSES_PATH"])
        )

        app.state.llm_classifier = llm_classifier
        app.state.bert_model = model_for_inference
        app.state.bert_tokenizer = tokenizer
        app.state.device = device

    except Exception as e:
        logging.error(f"Error in lifespan: {e}")
    yield


app = FastAPI(title="therapist_behaviour", version=__VERSION__,lifespan=lifespan)


app.include_router(api_router)

@app.get("/")
async def read_root()-> str:
    return f"The service is running in {STAGE} stage. version: {__VERSION__}"

if __name__ == "__main__":
    import uvicorn
    
    logging.basicConfig(level=logging.INFO)

    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=8000,
    )