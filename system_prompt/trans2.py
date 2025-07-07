import json
with open('/root/autodl-tmp/system_prompt/system_prompt_train_sampled.json', 'r') as f:
    data = json.load(f)

data_new = []
for i in range(0, len(data), 4):
    output = []
    for j in range(4):
        output.append(data[i+j]['output'][0])
    data_new.append({
        "prompt": data[i]['prompt'],
        "label": "",
        "output": output,
    })

with open('system_prompt_train_format.json', 'w') as f:
    json.dump(data_new, f, indent=4)