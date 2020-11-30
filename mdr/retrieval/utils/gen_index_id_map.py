import json

mapping = {}
with open('../data/para_doc.db') as f_in:
    for idx, line in enumerate(f_in):
        sample = json.loads(line.strip())
        mapping[idx] = sample['id']
with open('index_data/idx_id.json', 'w') as f_out:
    json.dump(mapping, f_out)

