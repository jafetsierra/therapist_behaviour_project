import logging
from fastapi import FastAPI
from transformers import DistilBertTokenizer
import torch

from config import ENV_VARIABLES

__VERSION__ = "0.1.0"
STAGE = ENV_VARIABLES['stage']

async def lifespan(app: FastAPI):
    try:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model_for_inference = CustomModel()  # Use your custom model class here
        model_for_inference.load_state_dict(torch.load('distilbert_finetuned.pth'))
        model_for_inference.eval()

        # Define the device for computation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_for_inference.to(device)
    except Exception as e:
        logging.error(f"Error in lifespan: {e}")


app = FastAPI(title="therapist_behaviour", version=__VERSION__,lifespan=lifespan)


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