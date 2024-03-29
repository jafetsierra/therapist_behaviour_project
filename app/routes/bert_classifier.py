import torch
from fastapi import APIRouter, Request, Depends, HTTPException

from app.schemas import ApiRequest, ApiResponse

router = APIRouter(
    prefix="/bert_classifier"
)

def get_bert_classifier(request: Request):
    return request.app.state.bert_model, request.app.state.bert_tokenizer, request.app.state.device

@router.post("/",response_model=ApiResponse)
async def bert_classifier_predict(request: ApiRequest, bert = Depends(get_bert_classifier)):

    try:
        model, tokenizer, device = bert
        text = request.text

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Move the tensors to the device of the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        predicted_class_index = probabilities.argmax(dim=1).item()

        return ApiResponse(predictions=predicted_class_index)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))