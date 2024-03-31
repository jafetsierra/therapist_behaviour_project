import openai

from langchain.chains import LLMChain
from typing import Dict, Any, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from .utils import load_llm, load_prompt
from config import ENV_VARIABLES

class LlmClassifier():

    artifacts: Optional[dict] = {}

    def __init__(self, 
                 chain_config: Dict[str,Any],
                 classes: str = None
                ):
        self.classifier_chain = self._load_chain(
            config=chain_config
        )
        self.classes = classes
        
    
    def _load_chain(self, config) -> LLMChain:
        llm_config = config.pop("llm")
        prompt_config = config.pop("prompt")
        _ = config.pop("_type")
        self.artifacts["model"] = llm_config["_type"]
        self.artifacts["temperature"] = 0
        prompt = load_prompt(prompt_config)
        
        llm = load_llm(llm_config, self.artifacts["temperature"])

        return LLMChain(
            llm=llm,
            prompt=prompt,
        )
        
    async def run(self, 
                  query: str
                ):
        
        inputs = {
            "human_input": query,
            "context": await self._get_similar_context(query),
            "classes": self.classes
        }

        return await self.classifier_chain.apredict(**inputs)
    
    async def _get_similar_context(self,input_text: str):
        openai_client = openai.Client(
                api_key=ENV_VARIABLES['OPENAI_API_KEY']
            )
        embedding_model = "text-embedding-3-small"

        client = AsyncQdrantClient(ENV_VARIABLES['QDRANT_HOST'], port=ENV_VARIABLES['QDRANT_PORT'])

        embbeding = openai_client.embeddings.create(input=input_text, model=embedding_model).data[0].embedding

        results = await client.search(
            collection_name=f"{ENV_VARIABLES['QDRANT_COLLECTION_NAME']}",
            search_params=models.SearchParams(hnsw_ef=128, exact=False),
            query_vector=embbeding,
            limit=10,
        )

        return '\n'.join([f"{x.payload['text']}->{x.payload['category']}" for x in results])
