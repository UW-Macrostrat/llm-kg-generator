from openai import AsyncOpenAI
import os
import asyncio
import time
from httpx import AsyncClient, HTTPError
import json
from arq.connections import RedisSettings
from prompts.vllm_prompts import SYSTEM_PROMPT, CONTEXT, PROMPT_ID
from enum import Enum
from pydantic import BaseModel
from typing import List
from wrapper_classes.weaviate_wrapper import WeaviateWrapper, WeaviateText

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_SETTINGS = RedisSettings(
    host=REDIS_HOST,     
    port=REDIS_PORT,      
    password=REDIS_PASSWORD,  
)

RESULT_ENDPOINT = os.getenv("RESULT_ENDPOINT")

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
MODEL_ID = f"{MODEL_NAME}_PROMPT_{PROMPT_ID}"

class RelationshipType(str, Enum):
    strat_name_to_lith = "stratigraphic unit has lithology of"
    lith_to_lith_type = "lithology has type of"
    att_grains = "lithology has grains of"
    att_color = "lithology has color of"
    att_bedform = "lithology has bedform of"
    att_sed_structure = "lithology has sedimentary structure of"

class Triplet(BaseModel):
    reasoning: str
    head: str
    tail: str
    relationship: RelationshipType
    
    def serialize(self):
        return {
            "src": self.head,
            "relationship_type": self.relationship.name,
            "dst": self.tail,
        }

class TripletList(BaseModel):
    reasoning: str
    triplets: List[Triplet]
    
class ParagraphResult(BaseModel):
    triplet_list: TripletList
    paragraph_data: WeaviateText
    
    def serialize(self):
        return {
            "text": {
                "preprocessor_id": self.paragraph_data.preprocessor_id,
                "paper_id": self.paragraph_data.paper_id,
                "hashed_text": self.paragraph_data.hashed_text,
                "weaviate_id": self.paragraph_data.weaviate_id,
                "paragraph_text": self.paragraph_data.paragraph,
            },
            "relationships": [
                triplet.serialize() for triplet in self.triplet_list.triplets
            ],
        }

async def startup(ctx: dict):
    # initialize connections to vLLM and Weaviate
    ctx["openai_client"] = AsyncOpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        timeout=None
    )   
    ctx["weaviate"] = WeaviateWrapper(f"http://{os.getenv('WEAVIATE_HOST')}:{os.getenv('WEAVIATE_PORT')}", os.getenv("WEAVIATE_API_KEY"))
    ctx["httpx_client"] = AsyncClient()
    
    # process prompts into chat template
    ctx["prompt"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    for example in CONTEXT:
        ctx["prompt"].append({"role": "user", "content": example[0]})
        ctx["prompt"].append({"role": "assistant", "content": example[1]})
    ctx["parser"] = TripletList.model_json_schema()

    # wait for vLLM server to start
    while True:
        try:
            response = await ctx["httpx_client"].get("http://127.0.0.1:8000/health")
            response.raise_for_status()
            break
        except HTTPError:
            await asyncio.sleep(5)

    # warmup request
    await ctx["openai_client"].chat.completions.create(
        model=MODEL_NAME,
        messages=ctx["prompt"], 
        temperature=0.0,
        extra_body={
            "guided_json": json.dumps(ctx["parser"]),
            "guided_decoding_backend": "outlines",
            "stop_token_ids": [128009] # need to add this since there is a bug with llama 3 tokenizer
        } 
    )

    print("WORKER: Ready to accept jobs.")
    
async def shutdown(ctx: dict):
    await ctx["openai_client"].close()
    await ctx["httpx_client"].aclose()

async def request_vllm(ctx: dict, paragraph_data: WeaviateText) -> ParagraphResult:
    messages = ctx["prompt"].copy() 
    messages.append({"role": "user", "content": paragraph_data.paragraph})

    llm_output = await ctx["openai_client"].chat.completions.create(
        model=MODEL_NAME,
        messages=messages, 
        temperature=0.0,
        extra_body={
            "guided_json": json.dumps(ctx["parser"]),
            "guided_decoding_backend": "outlines",
            "stop_token_ids": [128009] # need to add this since there is a bug with llama 3 tokenizer
        } 
    )
    
    # validate llm output, return None if no triplets were extracted
    try:
        validated_output = TripletList.model_validate_json(llm_output.choices[0].message.content)
        paragraph_result = ParagraphResult(triplet_list=validated_output, paragraph_data=paragraph_data)
        if not validated_output.triplets:
            return None
        return paragraph_result
    except Exception as e:
        print(f"WORKER: Error validating LLM output: {e}")
        return None

async def store_results(ctx: dict, output_list: list[ParagraphResult], run_metadata: dict):
    # convert results into json and post to an endpoint
    serialized_results = []
    for paragraph_output in output_list:
        if paragraph_output is not None:
            serialized_output = paragraph_output.serialize()
            serialized_results.append(serialized_output)
    
    # post to Macrostrat endpoint if any triplets have been extracted in this batch
    if serialized_results:
        await ctx["httpx_client"].post(
            RESULT_ENDPOINT, 
            headers={
                "Content-Type": "application/json"
            },
            content=json.dumps({
                "run_id" : run_metadata["run_id"],
                "extraction_pipeline_id" : run_metadata["pipeline_id"],
                "model_id": MODEL_ID,
                "results": serialized_results
            }, ensure_ascii=False).encode("ascii", errors="ignore").decode() # remove non ascii characters
        )
        
        

async def process_paragraphs(ctx: dict, paragraph_batch: list[str], run_metadata: dict):
    # pull paragraph text from Weaviate and extract triplets using the LLM
    tasks = []
    for paragraph_data in ctx["weaviate"].get_paragraphs_for_ids(paragraph_batch):
        task = request_vllm(ctx, paragraph_data)
        tasks.append(task)
    output_list = await asyncio.gather(*tasks)
    
    # serialize results for batch and store in Macrostrat endpoint
    await store_results(ctx, output_list, run_metadata)

class WorkerSettings:
    redis_settings = REDIS_SETTINGS
    functions = [process_paragraphs]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = 3

async def main():
    # demo run of worker on 2 paragraphs
    ctx = {}
    await startup(ctx)
    await process_paragraphs(ctx, ["00000085-2145-4b37-b963-8c80d21b6964", "955616ce-f846-4665-9c64-d4709a34680d"])
    await shutdown(ctx)

if __name__ == "__main__":
    asyncio.run(main())