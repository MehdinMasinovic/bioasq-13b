import json

with open('../data/BioASQ-training13b/training13b.json') as f:
    data = json.load(f)

for q in data['questions']:
    if 'exact_answer' in q:
        ea = q['exact_answer']
        if isinstance(ea, str):
            q['exact_answer'] = [[ea]]
        elif isinstance(ea, list):
            # If it's a flat list of strings, wrap each in a list
            if ea and all(isinstance(x, str) for x in ea):
                q['exact_answer'] = [[x] for x in ea]

with open('../data/BioASQ-training13b/training13b_fixed.json', 'w') as f:
    json.dump(data, f, indent=2)