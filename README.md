# LLM based kg-generator
The kg-generator uses LLMs to extract triplets from paragraphs. Triplets are made up of a head, tail, and relationship and can be used to create a knowledge graph.
The generator first extracts any possible triplets from the given paragraphs. 
Then, to filter down the retrieved triplets, a second pass is used to remove irrelevant information and force the triplets to conform to given structure (ie. all relationships must be found in a shortlist). 

### Status
- We have run LLMs with engineered prompts through a dataset
- The smaller LLM models have problems with accuracy, especially when attempting to filter out irrelevant triplet relationships.  
- More work can be done to testing different prompts, models, and chaining strategies. 
