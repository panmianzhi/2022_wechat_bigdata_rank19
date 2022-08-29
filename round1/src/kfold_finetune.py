import os
import json
from config import args

import logging
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from sklearn.model_selection import KFold

from models.finetune_model import FinetuneUniterModel
from tricks.FGM import FGM
from tricks.EMA import EMA
from data.category_id_map import CATEGORY_ID_LIST
from data.data_helper import MultiModalDataset
from util import setup_seed, setup_logging, build_optimizer, evaluate



def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(input_ids=batch['title_input'].cuda(),
                                                  attention_mask=batch['title_mask'].cuda(),
                                                  visual_feats=batch['frame_input'].cuda(),
                                                  visual_attention_mask=batch['frame_mask'].cuda(),
                                                  category_label=batch['label'].cuda())
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(fold, train_dataloader, val_dataloader):
    model = FinetuneUniterModel(len(CATEGORY_ID_LIST)).cuda() # 加载bert
    if args.pretrain_model_path is not None: # 加载预训练
        print(args.pretrain_model_path)
        model_prefix = args.pretrain_model_path.split('/')[-1].split('_')[0]
        restore_checkpoint(model, args.pretrain_model_path)

    optimizer, scheduler = build_optimizer(args, model)

    if args.num_gpus > 1:
        model = nn.DataParallel(model)

    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs

    ema = EMA(model, 0.999)
    ema.register()

    fgm = FGM(model)

    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            model.train()
            loss, accuracy, _, _ = model(input_ids=batch['title_input'].cuda(),
                                         attention_mask=batch['title_mask'].cuda(),
                                         visual_feats=batch['frame_input'].cuda(),
                                         visual_attention_mask=batch['frame_mask'].cuda(),
                                         category_label=batch['label'].cuda()) # loss, accuracy, pred_label_id, label
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()

            #fgm
            fgm.attack()
            loss_adv, _, _, _ = model(input_ids=batch['title_input'].cuda(),
                                      attention_mask=batch['title_mask'].cuda(),
                                      visual_feats=batch['frame_input'].cuda(),
                                      visual_attention_mask=batch['frame_mask'].cuda(),
                                      category_label=batch['label'].cuda())
            loss_adv = loss_adv.mean()
            loss_adv.backward()
            fgm.restore()

            optimizer.step()
            ema.update()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Fold {fold} Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

            # 4. validation
            if step % args.val_steps == 0:
                logging.info("Start Val")
                ema.apply_shadow()

                loss, results = validate(model, val_dataloader)
                logging.info("Val Done")

                results = {k: round(v, 4) for k, v in results.items()}
                logging.info(f"Fold {fold} Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

                mean_f1 = results['mean_f1']
                if mean_f1 > best_score:
                    best_score = mean_f1
                    if args.num_gpus > 1:
                        torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
                               f'{args.savedmodel_path}/{model_prefix}_fold_{fold}.bin')
                    else:
                        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'mean_f1': mean_f1},
                               f'{args.savedmodel_path}/{model_prefix}_fold_{fold}.bin')

                ema.restore()


def kfold_train_val(k=5):
    dataset = MultiModalDataset(args, args.labeled_annotation, args.labeled_zip_feats)
    kfold = KFold(n_splits=k, shuffle=True)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        train_dataloader = DataLoader(dataset,
                                      batch_size=args.batch_size,
                                      sampler=train_subsampler,
                                      drop_last=True,
                                      pin_memory=True,
                                      num_workers=args.num_workers)
        val_dataloader = DataLoader(dataset,
                                    batch_size=args.val_batch_size,
                                    sampler=val_subsampler,
                                    drop_last=False,
                                    pin_memory=True,
                                    num_workers=args.num_workers)
        train_and_validate(fold, train_dataloader, val_dataloader)


def restore_checkpoint(model_to_load, restore_name='BEST_EVAL_LOSS'):
    restore_bin = str(f'{restore_name}')
    state_dict = torch.load(restore_bin)['model']

    own_state = model_to_load.state_dict()
    for name, param in state_dict.items():
        name = name.replace('module.', '')
        if name not in own_state:
            logging.info(f'Skipped: {name}')
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
 #           logging.info(f"Successfully loaded: {name}")
        except:
            pass
            logging.info(f"Part load failed: {name}")


def main():
    setup_logging()
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)
    kfold_train_val(k=5)


if __name__ == '__main__':
    main()
