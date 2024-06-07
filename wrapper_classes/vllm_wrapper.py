
from openai import AsyncOpenAI
import asyncio
import httpx
import json
from pydantic import BaseModel
import logging


class VLLMWrapper:
    def __init__(self, model_name: str, schema: BaseModel) -> None:
        self.model_name = model_name
        self.client = AsyncOpenAI(
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
        )
        self.schema = schema
        self.json_schema = schema.model_json_schema()
        
    async def startup(self) -> None:
        # wait for vLLM server to start
        async with httpx.AsyncClient() as httpx_client:
            while True:
                try:
                    response = await httpx_client.get("http://127.0.0.1:8000/health")
                    response.raise_for_status()
                    break
                except httpx.HTTPError:
                    await asyncio.sleep(5)

        logging.info("VLLM server ready.")
    
    async def shutdown(self):
        await self.client.close()

    async def guided_generate(self, prompt: dict, constrained: bool = False) -> BaseModel:
        extra_body = {
            "stop_token_ids": [128001, 128009]  # need to add this since there is a bug with llama 3 tokenizer
        }

        if constrained:
            extra_body["guided_json"] = json.dumps(self.json_schema)
            extra_body["guided_decoding_backend"] = "lm-format-enforcer"

        llm_output = await self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt, 
            temperature=0.0,
            max_tokens=1024,
            extra_body=extra_body
        )   

        # validate llm output
        try:
            validated_output = self.schema.model_validate_json(llm_output.choices[0].message.content)
            return validated_output
        except ValueError:
            # retry with constrained encoding
            if not constrained:
                return await self.guided_generate(prompt, True)
            else:
                return None