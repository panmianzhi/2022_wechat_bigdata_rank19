import sys
sys.path.append('../')

from config import args
import json
from category_id_map import category_id_to_lv2id
import matplotlib.pyplot as plt

with open(args.labeled_annotation, 'r', encoding='utf8') as f:
    anns = json.load(f)

distribution = {}
for annotation in anns:
    lv2 = category_id_to_lv2id(annotation['category_id'])
    if lv2 in distribution:
        distribution[lv2] += 1
    else:
        distribution[lv2] = 0

distribution = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
labels, number = zip(*distribution)

plt.figure(figsize=(15,5))
plt.xticks([])
plt.bar(range(len(labels)), number)
plt.savefig('category_distribution.jpg')