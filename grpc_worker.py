import grpc
import workerserver_pb2
import workerserver_pb2_grpc


import os
import requests
import pandas as pd
import asyncio
import json
import logging

from prompts.command_r_prompts import PROMPT

GPU_ID = os.environ["GPU_ID"]
HOST_NAME = os.environ["HOST_NAME"]
MODEL_NAME = "LLM"

VALID_RELATIONSHIPS = ["stratigraphic unit has lithology of", "lithology has color of"]


class Worker(workerserver_pb2_grpc.WorkerServerServicer):
    async def ProcessParagraph(
        self,
        request: workerserver_pb2.ParagraphRequest,
        context: grpc.aio.ServicerContext,
    ) -> workerserver_pb2.ParagraphResponse:
        
        data = {
            "temperature": 0.0,
            "prompt": PROMPT.format(paragraph=request.paragraph),
        }

        response = requests.post(f"http://llm_backend_{HOST_NAME}_{GPU_ID}:8080/completion", json=data)

        try:
            response_json = json.loads(response.json()["content"])
        except:
            logging.info("Error: LLM output cannot be parsed.")
            return workerserver_pb2.ParagraphResponse(error=True)
        
        if isinstance(response_json, list) or not "triplets" in response_json:
            logging.info("Error: LLM output cannot be parsed.")
            return workerserver_pb2.ParagraphResponse(error=True)

        relationship_list = []
        
        for triplet in response_json["triplets"]:
            try:
                if triplet["relationship"] in VALID_RELATIONSHIPS:
                    relationship_list.append(
                        workerserver_pb2.Triplet(
                            head=triplet["head"],
                            tail=triplet["tail"],
                            relationship_type=triplet["relationship"],
                        )
                    )
            except:
                logging.info("Error: Triplet cannot be parsed.")
                continue

        return workerserver_pb2.ParagraphResponse(
            error=False, relationships=relationship_list, model_used=MODEL_NAME
        )

    async def Heartbeat(
        self,
        request: workerserver_pb2.StatusRequest,
        context: grpc.aio.ServicerContext,
    ):
        return workerserver_pb2.StatusResponse(status=True)


async def serve() -> None:
    server = grpc.aio.server()
    workerserver_pb2_grpc.add_WorkerServerServicer_to_server(Worker(), server)
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
