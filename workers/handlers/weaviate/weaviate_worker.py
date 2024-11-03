import os
import asyncio
import argparse
import httpx
import json
from workers.prompts.weaviate_prompts import SYSTEM_PROMPT, CONTEXT, PROMPT_ID
import logging
import sys
from workers.wrapper_classes.weaviate_wrapper import WeaviateWrapper, WeaviateText
from workers.wrapper_classes.worker_wrapper import Worker
from workers.wrapper_classes.vllm_wrapper import VLLMWrapper
from workers.handlers.weaviate.types import TripletList, ParagraphResult
from workers.handlers.utils.utils import dump_output
import workers.pb.job_manager_pb2 as pb

MANAGER_HOST = os.getenv("MANAGER_HOST")
RESULT_ENDPOINT = os.getenv("RESULT_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_VERSION = PROMPT_ID


async def startup(ctx: dict):
    # initialize connections to Weaviate
    ctx["weaviate"] = WeaviateWrapper(
        f"http://{os.getenv('WEAVIATE_HOST')}:{os.getenv('WEAVIATE_PORT')}",
        os.getenv("WEAVIATE_API_KEY"),
    )
    timeout = httpx.Timeout(30.0)
    ctx["httpx_client"] = httpx.AsyncClient(timeout=timeout)

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

    logging.info("ready to accept jobs.")


async def shutdown(ctx: dict):
    await ctx["httpx_client"].aclose()
    await ctx["vllm"].shutdown()


async def request_vllm(ctx: dict, paragraph_data: WeaviateText) -> ParagraphResult:
    messages = ctx["prompt"].copy()
    messages.append({"role": "user", "content": paragraph_data.paragraph})
    output = await ctx["vllm"].guided_generate(messages)
    if not output or not output.triplets:
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
        )
        .encode("ascii", errors="ignore")
        .decode()  # remove non ascii characters
    )

    if return_results:
        return output_json
    if serialized_results:
        response = await ctx["httpx_client"].post(RESULT_ENDPOINT, headers={"Content-Type": "application/json"}, content=output_json)
        response.raise_for_status()


async def process_paragraphs(
    ctx: dict,
    job_data: pb.WeaviateJob,
    run_metadata: dict,
    return_results: bool = False,
) -> dict | None:
    paragraph_batch = job_data.paragraph_ids

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


async def run_worker(worker_count: int = 2):
    ctx = {}
    await startup(ctx)
    worker = Worker(MANAGER_HOST, ctx=ctx)
    await worker.run_pool(process_paragraphs, worker_count)
    await shutdown(ctx)


# async def main(worker: bool):
#     logging.basicConfig(level=logging.INFO, stream=sys.stdout)
#     logging.getLogger("httpx").setLevel(logging.WARNING)

#     if worker:
#         run_worker()
#     else:
#         # demo run of worker on 2 paragraphs
#         ctx = {}
#         dummy_metadata = {"RunID": 0, "PipelineId": 0}
#         await startup(ctx)
#         output = await process_paragraphs(
#             ctx,
#             [
#                 "0f8ce52f-8f0e-4b58-a6a6-7515a9965526",
#                 "53947580-833f-4eb3-8413-efbbddfa890b",
#             ],
#             dummy_metadata,
#             True,
#         )
#         print(json.dumps(output, indent=4))
#         await shutdown(ctx)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--worker", action="store_true")

#     asyncio.run(main(**vars(parser.parse_args())))
