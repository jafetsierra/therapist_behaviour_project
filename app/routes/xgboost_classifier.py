from fastapi import APIRouter, Request, Depends, HTTPException

from app.schemas import ApiRequest, ApiResponse

router = APIRouter(
    prefix="/xgboost_classifier"
)

def get_xgboost_classifier(request: Request):
    return request.app.state.xgboost_classifier, request.app.state.xgboost_vectorizer

@router.post("/",response_model=ApiResponse)
async def llm_classifier_predict(request: ApiRequest, xgboost = Depends(get_xgboost_classifier)):
    encode_dict = {'question': 0, 'therapist_input': 1, 'reflection': 2, 'other': 3}

    try:
        text = request.text
        xgb_classifier, vectorizer = xgboost
        X = vectorizer.transform([text])

        response = xgb_classifier.predict(X)[0]
        decode_dict = {value: key for key, value in encode_dict.items()}

        return ApiResponse(predictions=decode_dict.get(response))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))