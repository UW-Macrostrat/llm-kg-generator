import grpc
import workerserver_pb2
import workerserver_pb2_grpc


import os
import requests
import pandas as pd
import asyncio
import json
import logging

from prompts import PROMPT

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
            "prompt": PROMPT.format(input=request.paragraph),
        }

        response = requests.post(f"llm_backend_{HOST_NAME}_{GPU_ID}:8080", json=data)

        try:
            response_json = json.loads(response.json()["content"])
        except:
            print("Error: LLM output cannot be parsed.")
            return workerserver_pb2.ParagraphResponse(status=False)

        if isinstance(response_json, list):
            print("Error: LLM output cannot be parsed.")
            return workerserver_pb2.ParagraphResponse(status=False)

        relationship_list = []
        for triplet in response_json:
            try:
                if triplet["relationship"] in VALID_RELATIONSHIPS:
                    relationship_list.append(
                        workerserver_pb2.Triplet(
                            head=triplet["head"],
                            tail=triplet["tail"],
                            relationship_type=triplet["relationship"],
                        )
                    )
            except KeyError:
                print("Error: Trplet cannot be parsed.")
                continue

        return workerserver_pb2.ParagraphResponse(
            status=True, relationships=relationship_list, model_used=MODEL_NAME
        )

    async def Heartbeat(
        self,
        request: workerserver_pb2.StatusRequest,
        context: grpc.aio.ServicerContext,
    ):
        return workerserver_pb2.StatusResponse(status=True)


async def serve() -> None:
    server = grpc.aio.server()

    worker = Worker()

    workerserver_pb2_grpc.add_WorkerServerServicer_to_server(worker, server)

    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)

    logging.info("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()
