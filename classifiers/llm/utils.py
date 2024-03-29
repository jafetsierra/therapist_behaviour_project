import logging
import yaml

from cloudpathlib import AnyPath
from typing import Any, Dict
from langchain_community.llms.bedrock import Bedrock
from langchain.llms.loading import load_llm_from_config
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts.loading import load_prompt_from_config
from langchain_openai import ChatOpenAI
from fastapi import HTTPException

def load_llm(config: dict, temperature: float = 0.0):
    """Load LLM from config

    Args:
        config (dict): LLM config

    Returns:
        BaseLLM | BaseChatModel: The loaded LLM
    """

    if config["_type"] == "openai-chat":

        _ = config.pop("_type")
        config["streaming"] = config.pop("stream", False)
        return ChatOpenAI(cache=False, **config)

    return load_llm_from_config(config)

def load_prompt(config: dict) -> BasePromptTemplate:
    """Load prompt from config

    Args:
        config (dict): Prompt config

    Returns:
        BasePromptTemplate: The loaded prompt
    """

    if config["_type"] == "chat":

        system_template = load_prompt_from_config(config["messages"][0]["prompt"])
        user_template = load_prompt_from_config(config["messages"][1]["prompt"])

        system_message_prompt = SystemMessagePromptTemplate(prompt=system_template)
        human_message_prompt = HumanMessagePromptTemplate(prompt=user_template)

        return ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

    return load_prompt_from_config(config)

def load_yaml_dict(path: AnyPath) -> Dict[str, Any]:
    """
    Reads a file from AnyPath (s3 or local) 
    and returns its content as a parsed YAML object.
    """
    try:
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        err_msg = f"Error loading YAML file {path}"
        logging.error(err_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=err_msg) from e

def load_txt(path: AnyPath) -> str:
    """
    Reads a file from AnyPath (s3 or local) 
    and returns its content as a string.
    """
    try:
        with open(path,'r') as f:
            return f.read()
    except Exception as e:
        err_msg = f"Error loading text file {path}"
        logging.error(err_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=err_msg) from e
