import json

with open('/root/autodl-tmp/lf_results/system_prompt_test_sampled/generated_predictions.jsonl', 'r') as f:
    all_data = f.readlines()
    all_data = [json.loads(x) for x in all_data]

with open('/root/autodl-tmp/system_prompt/system_prompt_test.json', 'r') as f:
    ori_data = json.load(f)

data = [
    {
        'prompt': y['instruction'],
        'label': '',
        'output': [y['input'] + '\n' + x['predict']],
    }
    for x, y in zip(all_data, ori_data)
]

with open('system_prompt_test_sampled.json', 'w') as f:
    json.dump(data, f, indent=4)