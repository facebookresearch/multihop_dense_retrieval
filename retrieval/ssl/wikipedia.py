import json

wikidata_path = "/private/home/xwhan/KBs/Wikidata"

def load_alias():
    title2names = json.load(open(os.path.join(wikidata_path, 'title2names.json')))
    print(f'load {len(title2names)} title alias...')
    return title2names