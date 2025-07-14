# llm_services.py
import os
import yaml
import json
import requests
from dotenv import load_dotenv
from typing import Dict, Any, List, Union

# Import SDKs
from groq import Groq
from openai import OpenAI
from config import SYSTEM_PROMT

load_dotenv()

CONFIG_PATH = "config.yaml"

def load_llm_config() -> List[Dict[str, Any]]:
    """Loads LLM configurations from the YAML file."""
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get("llm_options", [])
    except FileNotFoundError:
        print(f"Error: Configuration file '{CONFIG_PATH}' not found.")
        return []
    
AVAILABLE_LLMS = load_llm_config()

class LLMProvider:
    """
    A manager class to handle the initialization and caching of LLM clients.
    This class is instantiated once and reused throughout the application.
    """
    def __init__(self):
        self._clients = {} 
        print("LLMProvider initialized.")

    def _get_client(self, provider_name: str, api_key: str) -> Any:
        """
        Gets or creates a client for a given provider, using an internal cache.
        This is a private method, as users will interact via get_llm_response.
        """
        if provider_name in self._clients:
            return self._clients[provider_name]

        print(f"Creating new client for: {provider_name}")
        if provider_name == "groq":
            client = Groq(api_key=api_key)
        elif provider_name == "openai":
            client = OpenAI(api_key=api_key)
        else:
            client = "ollama_requests" 
        
        self._clients[provider_name] = client
        return client

    def get_llm_response(self, config: Dict[str, Any], prompt: str) -> Union[Dict, None]:
        """
        The main public method to get a response from an LLM based on its configuration.
        It efficiently reuses client objects.
        """
        provider = config.get("provider")
        model = config.get("model")
        api_key = os.environ.get(config["api_key_env"]) if config.get("api_key_env") else None

        try:
            client = self._get_client(provider, api_key)

            # if provider == "ollama":
            #     response = requests.post(
            #         f"{config.get('base_url', 'http://localhost:11434')}/api/generate",
            #         json={"model": model, "prompt": prompt, "system": SYSTEM_PROMT, "stream": False, "format": "json"},
            #         timeout=120
            #     )
            #     response.raise_for_status()
            #     return json.loads(response.json().get("response", "{}"))

            messages = [{"role": "system", "content": SYSTEM_PROMT}, {"role": "user", "content": prompt}]
            
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                stream=False,
                response_format={"type": "json_object"}
            )
            return json.loads(completion.choices[0].message.content)

        except Exception as e:
            print(f"Error during LLM call to {provider}: {e}")
            return None