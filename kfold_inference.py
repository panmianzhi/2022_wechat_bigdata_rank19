import os

import torch
from torch.utils.data import SequentialSampler, DataLoader
import torch.nn.functional as F

from config import args
from data.data_helper import MultiModalDataset
from data.category_id_map import lv2id_to_category_id, CATEGORY_ID_LIST
from models.finetune_model import FinetuneUniterModel


def inference():
    # 1. load data
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_feats, test_mode=True)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.num_workers)

    # 2. load model
    pred_logits = None
    for ckpt_name in os.listdir(args.savedmodel_path):
        if not ckpt_name.endswith('.bin'):
            continue

        factor = 1
        if ckpt_name.startswith('mac'):
            factor = 0.4
        elif ckpt_name.startswith('nezha'):
            factor = 0.325
        else:
            factor = 0.275

        print(f'{ckpt_name} factor: {factor}')
        if pred_logits is None:
            pred_logits = factor * forward(dataloader, ckpt_name) # [n, 200]
        else:
            pred_logits += factor * forward(dataloader, ckpt_name)

    predictions = torch.argmax(pred_logits, dim=1).cpu().tolist()

    # 4. dump results
    with open(args.test_output_csv, 'w') as f:
        for pred_label_id, ann in zip(predictions, dataset.anns):
            video_id = ann['id']
            category_id = lv2id_to_category_id(pred_label_id)
            f.write(f'{video_id},{category_id}\n')


def forward(dataloader, ckpt_name):
    model = FinetuneUniterModel(len(CATEGORY_ID_LIST)).cuda()
    checkpoint = torch.load(f'{args.savedmodel_path}/{ckpt_name}', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = torch.nn.parallel.DataParallel(model)
    model.eval()

    # 3. inference
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            logit = model(input_ids=batch['title_input'].cuda(),
                                  attention_mask=batch['title_mask'].cuda(),
                                  visual_feats=batch['frame_input'].cuda(),
                                  visual_attention_mask=batch['frame_mask'].cuda(),
                                  inference=True)
            predictions.append(logit)

    return torch.cat(predictions)



if __name__ == '__main__':
    inference()
