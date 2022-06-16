import os
import logging
import torch
import torch.nn as nn
from models.uniter import UniterPretraining
from models.optimization import BertAdam
from data.lxrt_dataset import create_dataloaders
from config import args

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s  %(filename)s  %(funcName)s  %(message)s')

LOSSES_NAME = ('MLM_LOSS', 'ITM_LOSS', 'MRFR_LOSS')

class PreTraining(object):

    def __init__(self):
        print(args.__dict__)

        self.model = UniterPretraining.from_pretrained(args.bert_dir)
        if args.num_gpus > 1:
            self.model = nn.DataParallel(self.model.cuda())

        self.train_loader, self.eval_loader = create_dataloaders(args)

    def forward(self, batch):
        loss, losses = self.model(input_ids=batch['title_input'].cuda(),
                                  attention_mask=batch['title_mask'].cuda(),
                                  masked_lm_labels=batch['title_mlm_label'].cuda(),
                                  visual_feats=batch['frame_input'].cuda(),
                                  visual_mask=batch['frame_mask'].cuda(),
                                  origin_feat=batch['origin_frame'].cuda(),
                                  feat_mask_label=batch['frame_mask_label'].cuda(),
                                  matched_label=batch['is_matched'].cuda())
        return loss, losses.detach().cpu()

    def train_epoch(self, optim, batch):
        '''
        :param optim:
        :param batch: list of InputFeatures
        :return:
        '''
        optim.zero_grad()
        loss, losses = self.forward(batch)
        if args.num_gpus > 1:
            loss = loss.mean()
            losses = losses.mean(0)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        optim.step()

        return loss.item(), losses.cpu().numpy()

    def val_batch(self, batch):
        with torch.no_grad():
            loss, losses = self.forward(batch)
            if args.num_gpus > 1:
                loss = loss.mean()
                losses = losses.mean(0)
        return loss.item(), losses.cpu().numpy()

    def train(self):
        batch_per_epoch = len(self.train_loader)
        t_total = int(batch_per_epoch * args.max_epochs)
        warmup_ratio = 0.05
        warmup_iters = int(t_total * warmup_ratio)
        logging.info(f'Batch per epoch: {batch_per_epoch}')
        logging.info(f'Total Iters: {t_total}')
        logging.info(f'Warm up Iters: {warmup_iters}')
        optim = BertAdam(self.model.parameters(), lr=args.pretrain_lr, warmup=warmup_ratio, t_total=t_total)

        for epoch in range(0, args.max_epochs):
            self.model.train()
            total_loss = 0.
            total_losses = 0.
            for i, batch in enumerate(self.train_loader):
                loss, losses = self.train_epoch(optim, batch)
                total_loss += loss
                total_losses += losses

            logging.info('Epoch%d: Training loss is %0.4f' % (epoch, total_loss / batch_per_epoch))
            losses_str = "losses are "
            for name, l in zip(LOSSES_NAME, total_losses):
                losses_str += "%s: %0.4f " % (name, l / batch_per_epoch)
            logging.info(losses_str)

            self.eval_epoch()
            self.save(f'uniter_Epoch{epoch}', self.model, optim)

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

    def save(self, name, model, optim):
        save_obj = {
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
        }
        if os.path.exists(args.savedmodel_path):
            torch.save(save_obj, os.path.join(args.savedmodel_path, "%s.pth" % name))
        else:
            torch.save(save_obj, "%s.pth" % name)


if __name__=="__main__":
    nvlpPreTrain = PreTraining()
    nvlpPreTrain.train()