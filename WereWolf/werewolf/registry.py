import os
from typing import Dict
import openai
from pydantic import BaseModel


class Registry(BaseModel):
    """Registry for storing and building classes."""

    name: str
    entries: Dict = {}
    translator_entries: Dict = {} 

    def register(self, keys: list):
        def decorator(cls):
            for key in keys:
                if key in self.entries:
                    if self.entries[key] == cls:
                        continue # Already registered with the same class, ignore.
                    raise ValueError(f"Key {key} is already registered with a different class.")
                self.entries[key] = cls
            return cls
        return decorator


    def build(self, type: str, **kwargs):
        if type not in self.entries:
            raise ValueError(
                f'{type} is not registered. Please register with the .register("{type}") method provided in {self.name} registry'
            )
        agent_params = {}

        if ("sft" in type.lower() or "makto" in type.lower() or "gpt" in type.lower() or "o1" in type.lower() or "marbo" in type.lower()) and "port" in kwargs:
            port = kwargs["port"]
            ip = kwargs.get("ip", None)
            if ip is None:
                client = openai.OpenAI(
                    api_key="EMPTY",
                    base_url=f"http://localhost:{port}/v1",
                )
            else:
                client = openai.OpenAI(
                    api_key="EMPTY",
                    base_url=f"http://{ip}:{port}/v1",
                )
            agent_params = {
                "client": client,
                "tokenizer": None,
                "llm": kwargs.get("llm", type),
                "temperature": kwargs["temperature"],
                "rate_limit": kwargs.get("rate_limit", 6.0),
            }

        elif (
            "gpt" in type.lower()
            or "o1" in type.lower()
            or "gpt" in str(kwargs.get("llm", "")).lower()
            or "o1" in str(kwargs.get("llm", "")).lower()
        ):
            # [Standard OpenAI API Support]
            client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("OPENAI_API_BASE"),
            )
            agent_params = {
                "client": client,
                "tokenizer": None,
                "llm": kwargs["llm"],
                "temperature": kwargs["temperature"],
                "rate_limit": kwargs.get("rate_limit", 6.0),
            }
        elif 'human' in type.lower():
            agent_params = {
                "client": None,
                "tokenizer": None,
                "llm": None,
                "temperature": 0
            }
        return type, agent_params

    def build_agent(self, type: str,
                    player_idx,
                    agent_param,
                    env_param,
                    log_file):
        
        if type not in self.entries:
            raise ValueError(
                f'{type} is not registered. Please register with the .register("{type}") method provided in {self.name} registry'
            )
        build_kwargs = {
            "client": agent_param["client"],
            "tokenizer": agent_param["tokenizer"],
            "llm": agent_param["llm"],
            "temperature": agent_param["temperature"],
            "log_file": log_file,
        }
        type_l = str(type).lower()
        if ("gpt" in type_l or "o1" in type_l) and "rate_limit" in agent_param:
            build_kwargs["rate_limit"] = agent_param["rate_limit"]
        return self.entries[type](**build_kwargs)

    def get_all_entries(self):
        return self.entries
