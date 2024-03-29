
from langchain.chains import LLMChain
from typing import Dict, Any, Optional

from .utils import load_llm, load_prompt

class LlmClassifier():

    artifacts: Optional[dict] = {}

    def __init__(self, 
                 chain_config: Dict[str,Any],
                 context : str = None,
                 classes: str = None
                ):
        self.classifier_chain = self._load_chain(
            config=chain_config
        )
        self.context = context
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
            "context": self.context,
            "classes": self.classes
        }

        return await self.classifier_chain.apredict(**inputs)
