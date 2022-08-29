import os
from config import args
import logging
import time
import torch
import torch.nn as nn

from models.finetune_model_v2 import FinetuneUniterModel
from models.two_stream_model import TwoStreamModel
from tricks.FGM import FGM
from tricks.swad import swa_utils
from tricks.swad import swad as swad_module
from data.category_id_map import CATEGORY_ID_LIST
from data.finetune_dataset import create_dataloaders

from util import setup_seed, setup_logging, evaluate, build_optimizer


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


def train_and_validate(use_swad=True):
    train_dataloader, val_dataloader = create_dataloaders(args)
    print(f'len train: {len(train_dataloader)}, len val: {len(val_dataloader)}')

    if not args.use_lxrt:
        model = FinetuneUniterModel(len(CATEGORY_ID_LIST)).cuda() # 加载bert
        if args.ckpt_file is not None:
            print("加载预训练")
            print(args.ckpt_file)
            restore_checkpoint(model, args.ckpt_file)
    else:
        model = TwoStreamModel(len(CATEGORY_ID_LIST)).cuda()
        if args.ckpt_file is not None:
            print("加载预训练")
            print(args.ckpt_file)
            restore_checkpoint(model.roberta, args.ckpt_file, replace_name='module.roberta.')
            print('============ roberta end ================================')
            restore_checkpoint(model.video_fc, args.ckpt_file, replace_name='module.roberta.video_fc.')
            print('============ video fc end ================================')
            restore_checkpoint(model.vit, args.ckpt_file, replace_name='module.roberta.encoder.')
            print('============ vit end ================================')

    
    model = nn.DataParallel(model)
    optimizer, scheduler = build_optimizer(args, model)
    
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    
    if use_swad:
        swad_algorithm = swa_utils.AveragedModel(model)
        swad = swad_module.LossValley(n_converge=3, n_tolerance=6, tolerance_ratio=0.3)

    fgm = FGM(model)
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            model.train()
            loss, accuracy, _, _ = model(input_ids=batch['title_input'].cuda(),
                                         attention_mask=batch['title_mask'].cuda(),
                                         visual_feats=batch['frame_input'].cuda(),
                                         visual_attention_mask=batch['frame_mask'].cuda(),
                                         category_label=batch['label'].cuda())
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
            optimizer.zero_grad()
            scheduler.step()
            
            if use_swad:
                swad_algorithm.update_parameters(model, step=step) # update averaged inner model in swad_algorithm

            step += 1
            
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

            # 4. validation
            if step % args.val_steps == 0:
                logging.info("Start Val")

                loss, results = validate(model, val_dataloader)
                logging.info("Val Done")

                results = {k: round(v, 4) for k, v in results.items()}
                logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")

                mean_f1 = results['mean_f1']
                if mean_f1 > best_score:
                    best_score = mean_f1
                    # torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'mean_f1': mean_f1},
                    #        f'{args.savedmodel_path}/epoch_{epoch}_mean_f1_{mean_f1}.bin')

                if use_swad:
                    swad.update_and_evaluate(swad_algorithm, val_loss=loss)
                    if hasattr(swad, "dead_valley") and swad.dead_valley:
                        logging.info("SWAD valley is dead -> early stop !")
                        break
                    
                    swad_algorithm = swa_utils.AveragedModel(model) # reset
    
    if use_swad:
        logging.info('start val final model')
        swad_algorithm = swad.get_final_model()
        loss, results = validate(swad_algorithm.module, val_dataloader)

        results = {k: round(v, 4) for k, v in results.items()}
        logging.info(f"loss {loss:.3f}, {results}")
        mean_f1 = results['mean_f1']
        if mean_f1 > best_score:
            best_score = mean_f1
            torch.save({'model_state_dict': swad_algorithm.module.module.state_dict(), 'mean_f1': mean_f1},
                   f'{args.savedmodel_path}/swad_mean_f1_{mean_f1}.bin')

        start = swad_algorithm.start_step
        end = swad_algorithm.end_step
        logging.info(f'[{start}-{end}] (N={swad_algorithm.n_averaged})')

        
def restore_checkpoint(model_to_load, restore_name='BEST_EVAL_LOSS', replace_name='module.'):
    state_dict = torch.load(restore_name)['model']

    own_state = model_to_load.state_dict()
    for name, param in state_dict.items():
        name = name.replace(replace_name, '')
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