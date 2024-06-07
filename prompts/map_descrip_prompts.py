PROMPT_ID = 0
SYSTEM_PROMPT = """You are a geological expert who is will be perfoming analysis on geological papers Your task is to identify and extract relationship triplets that match any of the following relationships:

    "lithology has type of"
    "lithology has grains of"
    "lithology has color of"
    "lithology has bedform of"
    "lithology has sedimentary structure of"
    
Each extracted relationship must exactly match a relationship in the list above.
    
For each identified relationship/triplet, provide a detailed explanation of your reasoning, and format your findings as a list of objects where each object contains the following keys:
{
    "reasoning": Detail your reasoning behind why the relationship found in the text matches one of the relationships in the shortlist.
    "head": Specify the subject of the relationship (should only be one stratigraphic unit or lithologic unit).
    "tail": Specify the object of the relationship (should only be one stratigraphic unit or lithologic unit).
    "relationship": Cite the specific relationship from the list above that you have identified. The relationship you cite must have an exact matching in the list above.
}

Your output must be in valid json and follow this format:

{
  "reasoning": Explain whether any relevant triplets are found in the text.
  "triplets": A list of triplets, should be empty if no relevant ones are found
}

If there are no relevant relationships found, return an empty list. Do not make assumptions about the text. 
"""

CONTEXT = [
    (
        """Find the relevant triplets in the following text. Your relationships must only include the lithologies and lithology attributes given below.
Here are relevant lithologies found in the text:
sand
shale
sandstone
limestone
dolomite
Here are relevant lithology attributes found in the text (attribute and attribute type):
argillaceous  lithology type
dolomitic lithology type

### Description: 

Portage Chute Formation - basal sandstone-shale member; limestone, dolomitic and argillaceous; Surprise Creek Formation - microcrystalline dolomite
""",
        """{
  "reasoning": "No grain size, color, or bedform attributes are explicitly mentioned. Found relationships that fit the specified models for lithology types.",
  "triplets": [
    {
      "reasoning": "'Dolomitic' specifies the type of the 'limestone'.",
      "head": "limestone",
      "tail": "dolomitic",
      "relationship": "lithology has type of"
    },
    {
      "reasoning": "'Argillaceous' specifies the type of the 'limestone'.",
      "head": "limestone",
      "tail": "argillaceous",
      "relationship": "lithology has type of"
    }
  ]
}""",
    ),
    (
        """Find the relevant triplets in the following text. Your relationships must only include the lithologies and lithology attributes given below.
Here are relevant lithologies found in the text:
shale
limestone
dolomite
slate
phyllite
marble
Here are relevant lithology attributes found in the text (attribute and attribute type):
massive 	bedform
fine	grains
gray	color
dark	color

### Description: Mostly massive dolomite, with a basal unit of gray to dark-gray limestone interbedded with limy shale, and an upper unit of fine-grained massive limestone with some marble; Pend Oreille and northern and central Stevens Counties. Three-fold division less evident in Colville area. Dolomite, with minor basal unit of interbedded limestone and phyllite in the Addy-Dunn Mountain area of Stevens County. Marble, dolomite, limestone, and limy slate in Hunters' district. Dolomitic marble in southern Stevens and northern Lincoln Counties. Middle Cambrian fossils near base in Metaline district, and Bathyuriscus-Elrathina fauna in lower unit in Leadpoint district. Phosphatic brachiopods in upper unit in Leadpoint district tentatively assigned to Middle and Upper Cambrian.""",
        """{
  "reasoning": "Found multiple relationships that fit the specified relationships.",
  "triplets": [
    {
      "reasoning": "'Massive' directly describes the bedform of 'dolomite'.",
      "head": "dolomite",
      "tail": "massive",
      "relationship": "lithology has bedform of"
    },
    {
      "reasoning": "'Gray' and 'dark-gray' directly describe the colors of 'limestone'.",
      "head": "limestone",
      "tail": "gray",
      "relationship": "lithology has color of"
    },
    {
      "reasoning": "'Gray' and 'dark-gray' directly describe the colors of 'limestone'.",
      "head": "limestone",
      "tail": "dark-gray",
      "relationship": "lithology has color of"
    },
    {
      "reasoning": "'Fine' directly describes the grain size of 'limestone', and 'massive' describes its bedform.",
      "head": "limestone",
      "tail": "fine",
      "relationship": "lithology has grains of"
    },
    {
      "reasoning": "'Fine' directly describes the grain size of 'limestone', and 'massive' describes its bedform.",
      "head": "limestone",
      "tail": "massive",
      "relationship": "lithology has bedform of"
    },
    {
      "reasoning": "'Dolomitic' describes the type of 'marble'.",
      "head": "marble",
      "tail": "dolomitic",
      "relationship": "lithology has type of"
    }
  ]
}""",
    ),
    (
        """Find the relevant triplets in the following text. Your relationships must only include the lithologies and lithology attributes given below.
Here are relevant lithologies found in the text:
sand
shale
sandstone
limestone
dolomite
tuff
Here are relevant lithology attributes found in the text (attribute and attribute type):
tuffaceous	lithology type

### Description: 

Limestone and dolomite, sandstone and shale; in part tuffaceous.""",
        """{
  "reasoning": "The attribute 'tuffaceous' is mentioned but not explicitly linked to a specific lithology in the text, hence no triplets can be definitively formed without assumptions.",
  "triplets": []
}""",
    ),
]

sample_description = """Lava flows, breccia, volcaniclastic and epiclastic rocks mostly of andesitic and dacitic composition; includes minor amounts of altered basaltic rocks. Joint surfaces and cavities commonly lined with hematite or montmorillonite clay, secondary silica minerals, zeolites, celadonite, or calcite. Andesite and dacite typically have plagioclase, hornblende, and clinopyroxene phenocrysts; some flows aphyric. Platy flow-jointing common. Age, mostly Oligocene; may include some rocks of early Miocene age. As shown, may include some rocks older than Oligocene, correlative with upper parts of unit Tea. One potassium-argon age of about 28 Ma on porphyritic hornblende andesite from Sheep Creek, southwest corner of Union County, indicates in part coeval with unit Tsf"""
