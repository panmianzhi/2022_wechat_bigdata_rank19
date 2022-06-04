import sys
sys.path.append('../')

import re
from config import args
import json
from category_id_map import category_id_to_lv2id
import matplotlib.pyplot as plt

anns = []
with open(args.labeled_annotation, 'r', encoding='utf8') as f1, \
        open(args.unlabeled_annotation, 'r', encoding='utf8') as f2:
    anns.extend(json.load(f1))
    anns.extend(json.load(f2))


'''
distribution = {}
title_len = 0
for annotation in anns:
    title_len += len(annotation['title'])
    lv2 = category_id_to_lv2id(annotation['category_id'])
    if lv2 in distribution:
        distribution[lv2] += 1
    else:
        distribution[lv2] = 0
mean_title_len = title_len / len(anns) # 32.54676

distribution = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
labels, number = zip(*distribution)

plt.figure(figsize=(15,5))
plt.xticks([])
plt.bar(range(len(labels)), number)
plt.savefig('category_distribution.jpg')
'''

pattern = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")

title_len = 0
asr_len = 0
ocr_len = 0

for annotation in anns:
    asr = annotation['asr']
    print(asr)
    print('------------------')
    print(re.sub(pattern, '', asr))
    print('=======================')

    title_len += len(annotation['title'])
    asr_len += len(annotation['asr'])
    for ocr in annotation['ocr']:
        ocr_len += len(ocr['text'])
print(f'mean title len: {title_len / len(anns)}\n mean asr len: {asr_len / len(anns)}\n mean ocr len: {ocr_len / len(anns)}')

# mean title len: 32.57292363636363
# mean asr len: 85.15406818181818
# mean ocr len: 97.38129272727272