from datasets import load_from_disk
from transformers import AutoTokenizer
import random
import json

ds = load_from_disk("/root/autodl-tmp/synthetic_gpts")

tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/t5-base')

n = len(ds)
print('length: ', n)

random.seed(42)
idx = random.sample(range(n), 15000+1000)
train_idx = idx[:15000]
test_idx = idx[-1000:]

train_ds = []
for i in train_idx:
    x = ds[i]
    system_prompt = x['system_prompt']
    system_prompt = [0 if j==-100 else j for j in system_prompt]
    # print(system_prompt)
    system_prompt = tokenizer.batch_decode([system_prompt], skip_special_tokens=True)[0]
    for j in range(4):
        y = {
            'instruction': system_prompt,
            'input': x['questions'][j][3:],
            'output': '',
        }
        train_ds.append(y)

test_ds = []
for i in test_idx:
    x = ds[i]
    system_prompt = x['system_prompt']
    system_prompt = [0 if j==-100 else j for j in system_prompt]
    # print(system_prompt)
    system_prompt = tokenizer.batch_decode([system_prompt], skip_special_tokens=True)[0]
    for j in range(4):
        y = {
            'instruction': system_prompt,
            'input': x['questions'][j][3:],
            'output': '',
        }
        test_ds.append(y)

with open('system_prompt_train.json', 'w') as f:
    json.dump(train_ds, f, indent=4)
with open('system_prompt_test.json', 'w') as f:
    json.dump(test_ds, f, indent=4)