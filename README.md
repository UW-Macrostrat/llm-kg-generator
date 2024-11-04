# LLM based kg-generator
The kg-generator uses LLMs to extract triplets from paragraphs. Triplets are made up of a head, tail, and relationship and can be used to create a knowledge graph. The application uses vLLM for its batched processing throughput and constrained decoding. The LLM output is in json and is forced to comply by a shortlist of relationships. Chain of thought is encoraged in the prompts and output format to help increase LLM reasoning abilities. 

## Application Overview

- **Input**: The application consumes text paragraphs from a coordinator.
- **Processing**: Utilizes vLLM to host local LLMs to extract relationship triplets from the text.
- **Output**: After processing each batch, the results are sent to a specified endpoint.

### Configuration

The Docker container relies on a `.env` file to securely manage environment variables. Ensure this file contains the necessary keys and hosts configurations for the Redis queue, Weaviate, and etc. See the example `sample.env` file for reference. 

### Details

The kg-generator has a modular structure for easy adaptability. Wrapper classes have been implemented to abstract operations with the coordinator and vLLM. The modular design allows for easy integration of new handlers that can process different data sources.

## Building and Running

### Building Image

To build the Docker image for the application, run the following command:

```bash
docker build -t llm_kg_generator -f docker/cosmos.Dockerfile .
```

This builds a docker image for use on Cosmos sytems. You can use this with the graph pipeline using docker compose. See the graph pipeline repo for more details.
