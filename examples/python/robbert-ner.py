#!/usr/bin/env python3

"""
This is an example using the DeepFrog NER model for Transformers, the DeepFrog software itself is not needed for
this example.

Automatically downloading the model is implied, so you can use this to get
started straight away.
"""

from transformers import AutoTokenizer, AutoModel, pipeline

modelname = "proycon/robbert-ner-cased-sonar1-nld"
tokenizer = modelname

#Load manually:
#tokenizer = Auto#Or use pipelines:Tokenizer.from_pretrained(modelname)
#model = AutoModel.from_pretrained(modelname)

#Or use pipelines:
nlp = pipeline('ner', model=modelname, tokenizer=modelname) #you can add ignore_labels=[] or in later versions: group=True
result = nlp("Amsterdam is de hoofdstad van Nederland, maar de regering zetelt in Den Haag.")
print(result)
result = nlp("Jan studeert aan de Universiteit van Amsterdam.")
print(result)


