import os
from config import args

import logging
import time
import torch
import torch.nn as nn
from models.finetune_model import ClassificationModel, FinetuneUniterModel
from tricks.FGM import FGM
from tricks.EMA import EMA
from data.category_id_map import CATEGORY_ID_LIST
from data.data_helper import create_dataloaders
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


def train_and_validate():
    train_dataloader, val_dataloader = create_dataloaders(args)

    # model = ClassificationModel(len(CATEGORY_ID_LIST)).cuda()
    model = FinetuneUniterModel(len(CATEGORY_ID_LIST)).cuda() # we use single-stream model

    if args.ckpt_file is not None:
        restore_checkpoint(model, args.ckpt_file)
    optimizer, scheduler = build_optimizer(args, model)
    if args.num_gpus > 1:
        model = nn.DataParallel(model)

    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs

    ema = EMA(model, 0.999)
    ema.register()

    # fgm = FGM(model)
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

            '''
            fgm.attack()
            loss_adv, _, _, _ = model(input_ids=batch['title_input'].cuda(),
                                      attention_mask=batch['title_mask'].cuda(),
                                      visual_feats=batch['frame_input'].cuda(),
                                      visual_attention_mask=batch['frame_mask'].cuda(),
                                      category_label=batch['label'].cuda())
            loss_adv = loss_adv.mean()
            loss_adv.backward()
            fgm.restore()
            '''

            optimizer.step()
            ema.update()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

            # 4. validation
            if step % 500 == 0:
                ema.apply_shadow()

                loss, results = validate(model, val_dataloader)
                results = {k: round(v, 4) for k, v in results.items()}
                logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

                mean_f1 = results['mean_f1']
                if mean_f1 > best_score:
                    best_score = mean_f1
                    if args.num_gpus > 1:
                        torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'mean_f1': mean_f1},
                               f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')
                    else:
                        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'mean_f1': mean_f1},
                               f'{args.savedmodel_path}/model_epoch_{epoch}_mean_f1_{mean_f1}.bin')

                ema.restore()


def restore_checkpoint(model_to_load, restore_name='BEST_EVAL_LOSS'):
    restore_bin = str(f'{args.savedmodel_path}/{restore_name}.pth')
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
            logging.info(f"Successfully loaded: {name}")
        except:
            pass
            logging.info(f"Part load failed: {name}")


def main():
    setup_logging()
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)
    train_and_validate()


if __name__ == '__main__':
    main()