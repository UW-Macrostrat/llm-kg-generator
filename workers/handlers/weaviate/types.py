from enum import Enum
from pydantic import BaseModel
from workers.wrapper_classes.weaviate_wrapper import WeaviateText


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
            "reasoning": self.reasoning,
        }


class TripletList(BaseModel):
    reasoning: str
    triplets: list[Triplet]


class ParagraphResult(BaseModel):
    triplet_list: TripletList
    paragraph_data: WeaviateText

    def serialize(self):
        return {
            "text": {
                "text_type": "weaviate_text",
                "preprocessor_id": self.paragraph_data.preprocessor_id,
                "paper_id": self.paragraph_data.paper_id,
                "hashed_text": self.paragraph_data.hashed_text,
                "weaviate_id": self.paragraph_data.weaviate_id,
                "paragraph_text": self.paragraph_data.paragraph,
            },
            "relationships": [triplet.serialize() for triplet in self.triplet_list.triplets],
        }
