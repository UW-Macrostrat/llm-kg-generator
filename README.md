# LLM based kg-generator
The kg-generator uses LLMs to extract triplets from paragraphs. Triplets are made up of a head, tail, and relationship and can be used to create a knowledge graph. The application uses vLLM for its batched processing throughput and constrained decoding. The LLM output is in json and is forced to comply by a shortlist of relationships. Chain of thought is encoraged in the prompts and output format to help increase LLM reasoning abilities. 

## Application Overview

- **Input**: The application consumes text paragraphs from a Redis queue.
- **Processing**: Utilizes vLLM to host local LLMs to extract relationship triplets from the text.
- **Output**: After processing each batch, the results are sent to a specified endpoint.

### Configuration

The Docker container relies on a `.env` file to securely manage environment variables. Ensure this file contains the necessary keys and hosts configurations for the Redis queue, Weaviate, and etc. See the example `.env` file for reference. 

## Building and Running

### Building Image

To build the Docker image for the application, run the following command:

```bash
docker build -t llm_kg_generator -f docker/arq.Dockerfile .
```

### Starting Redis Queue Worker

After the image has been built, you can run worker container:

```bash
docker run --rm --gpus all -it --env-file .env --ipc=host llm_kg_generator
```

This starts the redis queue worker which will be listening to the redis instance specified in your .env file. 

### Starting Demo Run

To start a test run over some sample paragraphs, run the worker container like so:
```bash
docker run --rm --gpus all -it --env-file .env --ipc=host llm_kg_generator test
```

This starts the vLLM instance and processes a batch of sample paragraphs before printing the output.
