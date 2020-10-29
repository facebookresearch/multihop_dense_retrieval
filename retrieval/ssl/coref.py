import spacy
import csv
import json
from tqdm import tqdm
nlp = spacy.load('en')
import neuralcoref
neuralcoref.add_to_pipe(nlp)

abstract_path = "/private/home/xwhan/data/hotpot/tfidf/abstracts.txt"

def annotate_doc(doc):
    ann = nlp(doc["text"])
    print(ann._.has_coref)
    print(ann._.coref_clusters)
    return

if __name__ == "__main__":
    docs = [json.loads(l) for l in tqdm(open(abstract_path).readlines())]
    import pdb; pdb.set_trace()
    annotate_doc(docs[1])
