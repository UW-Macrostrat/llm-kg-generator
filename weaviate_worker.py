import os
import asyncio
import argparse
import httpx
import json
from prompts.weaviate_prompts import SYSTEM_PROMPT, CONTEXT, PROMPT_ID
from enum import Enum
from pydantic import BaseModel
import logging
import sys
from wrapper_classes.weaviate_wrapper import WeaviateWrapper, WeaviateText
from wrapper_classes.worker_wrapper import Worker
from wrapper_classes.vllm_wrapper import VLLMWrapper

MANAGER_HOST = os.getenv("MANAGER_HOST")
RESULT_ENDPOINT = os.getenv("RESULT_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_VERSION = PROMPT_ID

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
    # initialize connections to Weaviate
    ctx["weaviate"] = WeaviateWrapper(f"http://{os.getenv('WEAVIATE_HOST')}:{os.getenv('WEAVIATE_PORT')}", os.getenv("WEAVIATE_API_KEY"))
    ctx["httpx_client"] = httpx.AsyncClient()
    
    # process prompts into chat template
    ctx["prompt"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    for example in CONTEXT:
        ctx["prompt"].append({"role": "user", "content": example[0]})
        ctx["prompt"].append({"role": "assistant", "content": example[1]})

    # wait for vLLM server to start
    ctx["vllm"] = VLLMWrapper(MODEL_NAME, TripletList)
    await ctx["vllm"].startup()

    logging.info("Ready to accept jobs.")
    
async def shutdown(ctx: dict):
    await ctx["httpx_client"].aclose()
    await ctx["vllm"].shutdown()

async def request_vllm(ctx: dict, paragraph_data: WeaviateText) -> ParagraphResult:
    messages = ctx["prompt"].copy() 
    messages.append({"role": "user", "content": paragraph_data.paragraph})
    output = await ctx["vllm"].guided_generate(messages)
    if not output.triplets:
        return None
    else:
        return ParagraphResult(triplet_list=output, paragraph_data=paragraph_data)

async def store_results(ctx: dict, output_list: list[ParagraphResult], run_metadata: dict, return_results: bool) -> dict | None:
    # convert results into json and post to an endpoint
    serialized_results = []
    for paragraph_output in output_list:
        if paragraph_output is not None:
            serialized_output = paragraph_output.serialize()
            serialized_results.append(serialized_output)
    
    # post to Macrostrat endpoint if any triplets have been extracted in this batch
    if return_results:
        return {
                    "run_id" : run_metadata["RunID"],
                    "extraction_pipeline_id" : run_metadata["PipelineId"],
                    "model_name": MODEL_NAME,
                    "model_version": MODEL_VERSION,
                    "results": serialized_results
                }
    else:
        if serialized_results:
            try:
                await ctx["httpx_client"].post(
                    RESULT_ENDPOINT, 
                    headers={
                        "Content-Type": "application/json"
                    },
                    content=json.dumps({
                        "run_id" : run_metadata["RunID"],
                        "extraction_pipeline_id" : run_metadata["PipelineId"],
                        "model_name": MODEL_NAME,
                        "model_version": MODEL_VERSION,
                        "results": serialized_results
                    }, ensure_ascii=False).encode("ascii", errors="ignore").decode() # remove non ascii characters
                )
            except httpx.HTTPError:
                logging.error("Failed to connect to result endpoint.")
                exit(1)
            
async def process_paragraphs(ctx: dict, paragraph_batch: list[str], run_metadata: dict, return_results: bool = False) -> dict | None:
    # pull paragraph text from Weaviate and extract triplets using the LLM
    tasks = []
    for paragraph_data in ctx["weaviate"].get_paragraphs_for_ids(paragraph_batch):
        task = request_vllm(ctx, paragraph_data)
        tasks.append(task)
    output_list = await asyncio.gather(*tasks)
    
    # serialize results for batch and store in Macrostrat endpoint
    result = await store_results(ctx, output_list, run_metadata, return_results)
    if return_results:
        return result

async def main(worker: bool):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)  
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if worker:
        ctx = {}
        await startup(ctx)
        worker = Worker(MANAGER_HOST, ctx=ctx)
        await worker.run_pool(process_paragraphs, 2)
        await shutdown(ctx)
    else:
        # demo run of worker on 2 paragraphs
        ctx = {}
        dummy_metadata = {"RunID": 0, "PipelineId": 0}
        await startup(ctx)
        output = await process_paragraphs(ctx, ["0f8ce52f-8f0e-4b58-a6a6-7515a9965526", "53947580-833f-4eb3-8413-efbbddfa890b"], dummy_metadata, True)
        print(json.dumps(output, indent=4))
        await shutdown(ctx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")

    asyncio.run(main(**vars(parser.parse_args())))