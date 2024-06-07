import os
import asyncio
import argparse
import httpx
import json
from prompts.map_descrip_prompts import SYSTEM_PROMPT, PROMPT_ID, sample_description
from enum import Enum
from pydantic import BaseModel
from typing import List
import logging
import sys
import pandas as pd
from wrapper_classes.worker_wrapper import Worker
from wrapper_classes.vllm_wrapper import VLLMWrapper
from devtools import pprint

MANAGER_HOST = os.getenv("MANAGER_HOST")
RESULT_ENDPOINT = os.getenv("RESULT_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_VERSION = PROMPT_ID

class RelationshipType(str, Enum):
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
    description: str
    
    def serialize(self):
        return {
            "text": {
                "paragraph_text": self.description,
            },
            "relationships": [
                triplet.serialize() for triplet in self.triplet_list.triplets
            ],
        }

async def startup(ctx: dict):
    ctx["httpx_client"] = httpx.AsyncClient()
    
    # process prompts into chat template
    ctx["prompt"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    ctx["liths"] = pd.read_csv("/vllm-workspace/prompts/liths.csv", header=None)
    ctx["lith_atts"] = pd.read_csv("/vllm-workspace/prompts/lith_atts.csv", header=None)

    # wait for vLLM server to start
    ctx["vllm"] = VLLMWrapper(MODEL_NAME, TripletList)
    await ctx["vllm"].startup()

    logging.info("Ready to accept jobs.")
    
async def shutdown(ctx: dict):
    await ctx["httpx_client"].aclose()
    await ctx["vllm"].shutdown()

async def request_vllm(ctx: dict, description: str) -> ParagraphResult:
    # dynamically generate prompt by inserting lith and lith atts matches into the context
    prompt = "Find the relevant triplets in the following text.\n"
    filtered_liths = ctx["liths"][ctx["liths"].iloc[:, 0].apply(lambda x: str(x) in description)]
    filtered_lith_atts = ctx["lith_atts"][ctx["lith_atts"].iloc[:, 0].apply(lambda x: str(x) in description)]
    if not filtered_liths.empty:
        prompt += "Here are relevant lithologies found in the text:\n"
        for _, row in filtered_liths.iterrows():
            prompt += row[0] + "\n"
    if not filtered_lith_atts.empty:
        prompt += "Here are relevant lithology attributes found in the text:\n"
        for _, row in filtered_lith_atts.iterrows():
            prompt += row[0] + "\t" + row[1] + "\n"
            
    # TODO: change? currently skipping descriptions with no matches
    if filtered_liths.empty and filtered_lith_atts.empty:
        return None
    
    messages = ctx["prompt"].copy() 
    messages.append({"role": "user", "content": prompt})
        
    output = await ctx["vllm"].guided_generate(messages)
    
    pprint(output)

    if not output.triplets:
        return None
    else:
        return ParagraphResult(triplet_list=output, description=description)

async def store_results(ctx: dict, output_list: list[str], run_metadata: dict, return_results: bool) -> dict | None:
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
            
async def process_descriptions(ctx: dict, description_batch: list[str], run_metadata: dict, return_results: bool = False) -> dict | None:
    # pull paragraph text from Weaviate and extract triplets using the LLM
    tasks = []
    for description in description_batch:
        task = request_vllm(ctx, description)
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
        await worker.run_pool(process_descriptions, 2)
        await shutdown(ctx)
    else:
        # demo run of worker on a description
        ctx = {}
        dummy_metadata = {"RunID": 0, "PipelineId": 0}
        await startup(ctx)
        output = await process_descriptions(ctx, [sample_description], dummy_metadata, True)
        print(json.dumps(output, indent=4))
        await shutdown(ctx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")

    asyncio.run(main(**vars(parser.parse_args())))