import os
import logging
import torch
import torch.nn as nn
from models.uniter import UniterPretraining
from models.create_optimizer import create_optimizer
from data.pretrain_dataset import create_dataloaders

from transformers import get_cosine_schedule_with_warmup
from config import args

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s  %(filename)s  %(funcName)s  %(message)s')

try:
    NUM_GPUS = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    print(f'Visible gpus: {NUM_GPUS}')
except:
    assert False, 'cannot find gpus'

LOSSES_NAME = ('MLM_LOSS', 'ITM_LOSS', 'MRFR_LOSS')
LR = {'others':5e-4, 'roberta':5e-5}
LR_LAYER_DECAY = 1.0
WARMUP_RATIO = 0.06


class PreTraining(object):

    def __init__(self):
        print(args.__dict__)

        self.model = UniterPretraining()
        if NUM_GPUS > 1:
            self.model = nn.DataParallel(self.model)

        self.train_loader, self.eval_loader = create_dataloaders(args)

    def forward(self, batch):
        
        if NUM_GPUS==0:
            loss, losses = self.model(title_ids=batch['title_input'],
                              title_mask=batch['title_mask'],
                              masked_lm_labels=batch['title_mlm_label'],
                              visual_feats=batch['frame_input'],
                              visual_mask=batch['frame_mask'],
                              origin_feat=batch['origin_frame'],
                              mfm_label=batch['frame_mask_label'],
                              matched_label=batch['is_matched'])
        else:                        
            loss, losses = self.model(title_ids=batch['title_input'].cuda(),
                                      title_mask=batch['title_mask'].cuda(),
                                      masked_lm_labels=batch['title_mlm_label'].cuda(),
                                      visual_feats=batch['frame_input'].cuda(),
                                      visual_mask=batch['frame_mask'].cuda(),
                                      origin_feat=batch['origin_frame'].cuda(),
                                      mfm_label=batch['frame_mask_label'].cuda(),
                                      matched_label=batch['is_matched'].cuda())
                                  
        return loss, losses.detach().cpu()

    def train_epoch(self, optim, scheduler, batch):
        '''
        :param batch: list of InputFeatures
        :return:
        '''
        optim.zero_grad()
        loss, losses = self.forward(batch)

        loss = loss.mean()
        losses = losses.mean(0)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        optim.step()
        scheduler.step()

        return loss.item(), losses.cpu().numpy()

    def val_batch(self, batch):
        with torch.no_grad():
            loss, losses = self.forward(batch)

            loss = loss.mean()
            losses = losses.mean(0)
        return loss.item(), losses.cpu().numpy()

    def train(self):
        optimizer = create_optimizer(self.model, model_lr=LR, layerwise_learning_rate_decay=LR_LAYER_DECAY)

        batch_per_epoch = len(self.train_loader)
        total_steps = args.max_epochs * batch_per_epoch
        warmup_steps = int(WARMUP_RATIO * total_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_steps)

        best_loss = None
        for epoch in range(args.max_epochs):
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                loss, losses = self.train_epoch(optimizer, scheduler, batch)
                if i % 1000 == 0:
                    losses_str = f"Epoch{epoch}, step {i}: Training loss is {loss}. "
                    for name, l in zip(LOSSES_NAME, losses):
                        losses_str += "%s: %0.4f " % (name, l)
                    logging.info(losses_str)

            eval_loss = self.eval_epoch()
            if (not best_loss) or eval_loss < best_loss:
                best_loss = eval_loss
                self.save(args.pretrain_model_name, self.model, optimizer, scheduler)
                
            if epoch % 5 == 0:
                self.save(f'{args.pretrain_model_name}_{epoch}', self.model, optimizer, scheduler)

    def eval_epoch(self):
        self.model.eval()
        total_loss = 0.
        total_losses = 0.
        for i, batch in enumerate(self.eval_loader):
            loss, losses = self.val_batch(batch)
            total_loss += loss
            total_losses += losses

        logging.info('Eval loss is %0.4f' % (total_loss / len(self.eval_loader)))
        losses_str = "losses are "
        for name, l in zip(LOSSES_NAME, total_losses / len(self.eval_loader)):
            losses_str += "%s: %0.4f " % (name, l)
        logging.info(losses_str)

        return total_loss / len(self.eval_loader)

    def save(self, name, model, optim, scheduler):
        model_obj = {
            'model': model.state_dict()
        }
        optim_obj = {
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict()
        }

        torch.save(model_obj, os.path.join(args.savedmodel_path, f'{name}_model.pth'))
        torch.save(optim_obj, os.path.join(args.savedmodel_path, f'{name}_optim.pth'))


if __name__=="__main__":
    os.makedirs(args.savedmodel_path, exist_ok=True)
    nvlpPreTrain = PreTraining()
    nvlpPreTrain.train()