from enum import Enum
from pydantic import BaseModel
from typing import List


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
            "reasoning": self.reasoning,
        }


class TripletList(BaseModel):
    reasoning: str
    triplets: List[Triplet]


class ParagraphResult(BaseModel):
    triplet_list: TripletList
    description: str
    prompt: str
    legend_id: int

    def serialize(self):
        return {
            "text": {
                "text_type": "map_descriptions",
                "paragraph_text": self.description,
                "legend_id": self.legend_id,
            },
            "relationships": [triplet.serialize() for triplet in self.triplet_list.triplets],
        }
