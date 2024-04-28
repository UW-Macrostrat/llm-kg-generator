import os
from dataclasses import dataclass
from typing import Protocol, Iterable, Any, Callable
import weaviate
import json

@dataclass
class WeaviateText:
    preprocessor_id: str
    paper_id: str
    hashed_text: str
    weaviate_id: str
    paragraph: str

class WeaviateWrapper:

    def __init__(self, endpoint_url : str, api_key : str):
        self.client = weaviate.Client(
            endpoint_url,
            auth_client_secret=weaviate.auth.AuthApiKey(api_key),
        )
    
    def get_paragraphs_for_ids(self, ids_to_load : "Iterable[str]") -> "Iterable[WeaviateText]":
        for paragraph_id in ids_to_load:
            # Make the query
            response = (
                self.client.query
                .get("Paragraph", ['preprocessor_id', 'paper_id', 'hashed_text', 'text_content'])
                .with_where({
                    "path": "id",
                    "operator": "Equal",
                    "valueText": paragraph_id
                })
                .do()
            )

            # Ensure the result exists
            if "data" not in response or "Get" not in response["data"] or "Paragraph" not in response["data"]["Get"]:
                continue
            
            # Return the result
            paragraph_data = response["data"]["Get"]["Paragraph"][0]
            yield WeaviateText(
                preprocessor_id = paragraph_data["preprocessor_id"],
                paper_id = paragraph_data["paper_id"],
                hashed_text = paragraph_data["hashed_text"],
                weaviate_id = paragraph_id,
                paragraph = paragraph_data["text_content"]
            )
            
def main():
    weaviate_wrapper = WeaviateWrapper("http://cosmos0001.chtc.wisc.edu:8080", os.getenv("HYBRID_API_KEY"))
    for result in weaviate_wrapper.get_paragraphs_for_ids(["00000085-2145-4b37-b963-8c80d21b6964"]):
        print(result)

if __name__ == "__main__":
    main()