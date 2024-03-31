from fastapi import APIRouter, Request, Depends, HTTPException

from app.schemas import ApiRequest, ApiResponse

router = APIRouter(
    prefix="/llm_classifier"
)

def get_llm_classifier(request: Request):
    return request.app.state.llm_classifier

@router.post("/",response_model=ApiResponse)
async def llm_classifier_predict(request: ApiRequest, llm = Depends(get_llm_classifier)):

    try:
        text = request.text

        response = await llm.run(query=text)
        
        return ApiResponse(predictions=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))