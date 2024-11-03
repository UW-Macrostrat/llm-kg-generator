import os
import asyncio
import argparse
import httpx
import json
from workers.prompts.map_descrip_prompts import (
    SYSTEM_PROMPT,
    CONTEXT,
    PROMPT_ID,
    sample_description,
)
import logging
import sys
import pandas as pd
import re

from workers.wrapper_classes.vllm_wrapper import VLLMWrapper
from workers.handlers.map_descriptions.types import TripletList, ParagraphResult
import workers.pb.job_manager_pb2 as pb
from workers.handlers.utils.utils import dump_output


MANAGER_HOST = os.getenv("MANAGER_HOST")
RESULT_ENDPOINT = os.getenv("RESULT_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_VERSION = PROMPT_ID


async def startup(ctx: dict):
    timeout = httpx.Timeout(30.0)
    ctx["httpx_client"] = httpx.AsyncClient(timeout=timeout)

    # process prompts into chat template
    ctx["prompt"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    for example in CONTEXT:
        ctx["prompt"].append({"role": "user", "content": example[0]})
        ctx["prompt"].append({"role": "assistant", "content": example[1]})
    ctx["liths"] = pd.read_csv("/vllm-workspace/workers/prompts/liths.csv", header=None)
    ctx["lith_atts"] = pd.read_csv("/vllm-workspace/workers/prompts/lith_atts.csv", header=None)

    # wait for vLLM server to start
    ctx["vllm"] = VLLMWrapper(MODEL_NAME, TripletList)
    await ctx["vllm"].startup()

    logging.info("Ready to accept jobs.")


async def shutdown(ctx: dict):
    await ctx["httpx_client"].aclose()
    await ctx["vllm"].shutdown()


def match_word(word, description):
    return bool(re.search(r"\b" + re.escape(str(word)) + r"(\b|\.|,)", description.lower()))


async def generate_triplets(ctx: dict, map_description: pb.MapDescriptionJob) -> ParagraphResult:
    description = map_description.text

    # dynamically generate prompt by inserting liths and lith atts matches into the context
    prompt = "Find the relevant triplets in the following text."
    prompt += "Your relationships must only include the lithologies and lithology attributes given below.\n"
    filtered_liths = ctx["liths"][ctx["liths"].iloc[:, 0].apply(lambda x: match_word(x, description))]
    filtered_lith_atts = ctx["lith_atts"][ctx["lith_atts"].iloc[:, 0].apply(lambda x: match_word(x, description))]
    # TODO: change? currently skipping descriptions with no matches
    if filtered_liths.empty or filtered_lith_atts.empty:
        return None
    prompt += "Here are relevant lithologies found in the text:\n"
    for _, row in filtered_liths.iterrows():
        prompt += row[0] + "\n"
    prompt += "Here are relevant lithology attributes found in the text (attribute and attribute type):\n"
    for _, row in filtered_lith_atts.iterrows():
        prompt += row[0] + "\t" + row[1] + "\n"
    prompt += "\n"
    prompt += "###\n"
    prompt += description

    messages = ctx["prompt"].copy()
    messages.append({"role": "user", "content": prompt})

    output = await ctx["vllm"].guided_generate(messages)
    if not output or not output.triplets:
        return None
    return ParagraphResult(triplet_list=output, description=description, prompt=prompt, legend_id=map_description.legend_id)


async def store_results(ctx: dict, output_list: list[ParagraphResult], run_metadata: dict, return_results: bool) -> dict | None:
    # convert results into json and post to an endpoint
    serialized_results = []
    for paragraph_output in output_list:
        if paragraph_output is not None:
            serialized_output = paragraph_output.serialize()
            serialized_results.append(serialized_output)

    # post to Macrostrat endpoint if any triplets have been extracted in this batch
    output_json = (
        json.dumps(
            {
                "run_id": run_metadata.run_id,
                "extraction_pipeline_id": run_metadata.pipeline_id,
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "results": serialized_results,
            },
            ensure_ascii=False,
            indent=4,
        )
        .encode("ascii", errors="ignore")
        .decode()  # remove non ascii characters
    )

    if return_results:
        return output_json
    if serialized_results:
        # dump_output(output_json)
        print(output_json)
        response = await ctx["httpx_client"].post(RESULT_ENDPOINT, headers={"Content-Type": "application/json"}, content=output_json)
        print(response.text)
        response.raise_for_status()


async def process_descriptions(
    ctx: dict,
    job_data: pb.MapDescriptionJob,
    run_metadata: dict,
    return_results: bool = False,
) -> dict | None:
    description_batch = job_data.descriptions

    # pull paragraph text from Weaviate and extract triplets using the LLM
    tasks = []
    for description in description_batch:
        task = generate_triplets(ctx, description)
        tasks.append(task)
    output_list = await asyncio.gather(*tasks)

    # serialize results for batch and store in Macrostrat endpoint
    result = await store_results(ctx, output_list, run_metadata, return_results)
    if return_results:
        return result


# async def run_worker(worker_count: int = 2):
#     ctx = {}
#     await startup(ctx)
#     worker = Worker(MANAGER_HOST, ctx=ctx)
#     await worker.run_pool(process_descriptions, worker_count)
#     await shutdown(ctx)


# async def main(worker: bool):
#     logging.basicConfig(level=logging.INFO, stream=sys.stdout)
#     logging.getLogger("httpx").setLevel(logging.WARNING)

#     if worker:
#         run_worker()
#     else:
#         # demo run of worker on a description
#         ctx = {}
#         dummy_metadata = {"RunID": 0, "PipelineId": 0}
#         await startup(ctx)
#         output = await process_descriptions(ctx, [sample_description], dummy_metadata, True)
#         print(json.dumps(output, indent=4))
#         await shutdown(ctx)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--worker", action="store_true")

#     asyncio.run(main(**vars(parser.parse_args())))
