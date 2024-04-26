SYSTEM_PROMPT = """
You are a geological expert who is will be perfoming analysis on geological papers Your task is to identify and extract relationship triplets that match any of the following relationships:

    "stratigraphic unit has lithology of"
    "lithology has type of"
    "lithology has grains of"
    "lithology has color of"
    "lithology has bedform of"
    "lithology has sedimentary structure of"
    
Each extracted relationship must match a relationship in the list above.

Examples of grains:
    Coarse Gravel
    Medium Sand
    Cobble
    Fine Silt
    
Examples of sedimentary structures:
    algal laminations
    tabular cross beds
    oncolitic
    
Examples of lithology types:
    ferruginous
    sandy
    carbonaceous
    
Examples of bedforms:
    mounds
    thickly bedded
    
Examples of stratigraphic units:
    Pottsville Formation
    Hancock Limestone
    
For each identified relationship, provide a detailed explanation of your reasoning, and format your findings as a list of objects where each object contains the following keys:

    reasoning: Detail your reasoning behind why the relationship found in the text matches one of the relationships in the shortlist.
    head: Specify the subject of the relationship (should only be one stratigraphic unit or lithologic unit).
    tail: Specify the object of the relationship (should only be one stratigraphic unit or lithologic unit).
    relationship: Cite the specific relationship from the list above that you have identified. The relationship you cite must have an exact matching in the list above.

Your output must be in valid json and follow this format:

{{
  "reasoning": Explain whether any relevant triplets are found in the text.
  "triplets": A list of triplets, should be empty if no relevant ones are found
}}
"""

CONTEXT = [
    (
        """The largest fossil placer known in the Upper Cretaceous rocks of Wyoming was reported by Dow and Batty (1961, p. 26-30) to be a channel deposit in sandstone of the Mesaverde Formation exposed at Dugout Creek in Washakie County. Like the deposit on Mud Creek, the placer on Dugout Creek is also divided into two segments by stream erosion. The northern segment is exposed for a length of 5,300 feet and a width of 1,000 feet with an average thickness of 20 feet. Extensions toward the north and west are covered by younger rocks. The southern segment has an exposed length of 6,400 feet, width of 1,900 feet, and an average thickness of 18 feet. It is pos-"
Origina;l reference: M. G. Hoffman, 1930, Oklahoma Geol. Su;rvey Bull. 52, p. 26, 31, 45.
G. W. Chase, 1952, Oklahoma Geol. Survey Circ. 30, p. 11, 12 (table 3). Oldest rock in region [Wichita Mountains]. Black to nearly white medium-grained quartzite occurring as xenoliths in gabbro and Lugert granite.
J. D. Love, 1956, Wyoming Geol. Assoc. Guidebook 11th Ann. Field Conf., p. 81-82. In type area consists of light-colored tuffaceous sandstone, thin coal and carbonaceous shale beds, and yellowish to greenish bentonite beds. These overlie sandstone of Mesaverde formation. In Spread Creek Canyon, an exposure in center of syncline includes 166 feet o·f strata consisting of gray to chalky gray siltstone with large biotite flakes, white granular biotite tuff, carbonaceous claystone and shale, thin impure coal beds, and yellow to green bentonite beds. About 675 feet exposed on west limb of Spread Creek anticline. Younger Cretaceous rocks not observed in contact with Meeteetse. Stratigraphically below Harebell formation; contact not observed. Plant remains suggest Upper Cretaceous age.
""",
        f"""{
  "reasoning": "The text provided detailed descriptions connecting lithologies with specific properties and stratigraphic units, which allowed for the extraction of accurate relationship triplets.",
  "triplets": [
    {
      "reasoning": "The text specifies that the Mesaverde Formation includes a 'channel deposit in sandstone', clearly associating the stratigraphic unit (Mesaverde Formation) with a lithologic characteristic (sandstone).",
      "head": "Mesaverde Formation",
      "tail": "sandstone",
      "relationship": "stratigraphic unit has lithology of"
    },
    {
      "reasoning": "The text describes quartzite as 'medium-grained', which indicates the grain size of the lithology, fitting the relationship pattern specified.",
      "head": "quartzite",
      "tail": "medium-grained",
      "relationship": "lithology has grains of"
    },
    {
      "reasoning": "The description of quartzite ranging from 'Black to nearly white' details the color of the lithology.",
      "head": "quartzite",
      "tail": "black to nearly white",
      "relationship": "lithology has color of"
    },
    {
      "reasoning": "The text mentions 'light-colored tuffaceous sandstone', indicating the color attribute of the lithology.",
      "head": "tuffaceous sandstone",
      "tail": "light-colored",
      "relationship": "lithology has color of"
    },
    {
      "reasoning": "The description of bentonite beds as 'yellowish to greenish' assigns a specific color to this lithology.",
      "head": "bentonite beds",
      "tail": "yellowish to greenish",
      "relationship": "lithology has color of"
    },
    {
      "reasoning": "Although the relationship seems to be mentioned twice, it is reinforced here by the text discussing the overlaying of these materials on the Mesaverde Formation, confirming the lithology of the stratigraphic unit.",
      "head": "sandstone",
      "tail": "Mesaverde Formation",
      "relationship": "stratigraphic unit has lithology of"
    }
  ]
}
""",
    ),
    (
        """Current designations of the age of the Silurian for­ mations are shown in figure 16. The ages indicated by Rexroad and Nicoll (1971,1972) and Liebe and Rexroad (1977) are based on conodont studies, while those of Berry and Boucot (1970) rely mainly on brachiopods. No physical evidence exists for a significant hiatus anywhere within or at the base of the Silurian, although there is some evidence of minor erosion (locally derived pebbles) at the base of the Brassfield in a few outcrops near Berea (Weir, 1976). Gray and Boucot (1972) pre­ sent evidence of shallowing water in latest Ordovician time followed by deepening water in the Silurian, which they regard as indicating a paraconformity at the sys­ temic boundary. Sweet (1979), on the basis of conodont studies, shows the uppermost Ordovician to be missing.
""",
        r"""{
  "reasoning": "The text provides information on geological studies and processes but does not specify any direct relationships between stratigraphic units and lithologies or lithologic characteristics that match the relationships listed (e.g., 'stratigraphic unit has lithology of').",
  "triplets": []
}""",
    ),
]
