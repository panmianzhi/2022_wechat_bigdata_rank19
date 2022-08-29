import sys
sys.path.append('../')
from config import args

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .uniter_v2 import UniModel # 主要修改这里


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class FinetuneUniterModel(nn.Module):

    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class
        self.roberta = UniModel.from_pretrained(args.bert_dir, img_dim=args.frame_embedding_size)
        hid_dim = self.roberta.config.hidden_size

        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            nn.LayerNorm(hid_dim  * 2, eps=1e-12),
            nn.Linear(hid_dim  * 2, num_class)
        )
        # init_weights_b(self.logit_fc)
        self.logit_fc.apply(self.roberta._init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                visual_feats=None, visual_attention_mask=None,
                category_label=None, inference=False):
        
        last_hidden_states = self.roberta(input_ids, attention_mask, visual_feats, visual_attention_mask)

        logit = self.logit_fc(last_hidden_states[:, 0, :])

        if inference:
            return logit
        else:
            return self.cal_loss(logit, category_label)

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


# following are model for single-model semi-supervised finetune
class SingleModel_a(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class
        self.roberta = UniModel.from_pretrained(args.bert_dir, img_dim=args.frame_embedding_size)
        hid_dim = self.roberta.config.hidden_size

        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            nn.LayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_class)
        )
        self.logit_fc.apply(self.roberta._init_weights)

    def forward(self, batch, inference=False):
        last_hidden_states = self.roberta(input_ids=batch['title_input'], 
                                          text_attention_mask=batch['title_mask'], 
                                          video_feat=batch['frame_input'], 
                                          video_attention_mask=batch['frame_mask'])
        logit = self.logit_fc(last_hidden_states[:, 0, :])
        
        if inference:
            return torch.argmax(logit, dim=1)
        else:
            return logit
        

class SingleModel_b(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class
        self.roberta = UniModel.from_pretrained(args.bert_dir, img_dim=args.frame_embedding_size)
        hid_dim = self.roberta.config.hidden_size

        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            nn.LayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_class)
        )
        init_weights_b(self.logit_fc)

    def forward(self, batch, inference=False):
        last_hidden_states = self.roberta(input_ids=batch['title_input'], 
                                          text_attention_mask=batch['title_mask'], 
                                          video_feat=batch['frame_input'], 
                                          video_attention_mask=batch['frame_mask'])
        logit = self.logit_fc(last_hidden_states[:, 0, :])
        
        if inference:
            return torch.argmax(logit, dim=1)
        else:
            return logit
        

class MultiModel(nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def forward(self, inputs, unlabel_inputs):

        output_a = self.model_a(inputs)
        output_b = self.model_b(inputs)

        un_output_a = self.model_a(unlabel_inputs)
        un_output_b = self.model_b(unlabel_inputs)

        un_gt_a = torch.argmax(un_output_b, dim=1).unsqueeze(1)
        un_gt_b = torch.argmax(un_output_a, dim=1).unsqueeze(1)

        loss_a, accuracy_a, pred_label_id_a, label_a = self.cal_loss(output_a, inputs['label'])
        loss_b, accuracy_b, pred_label_id_b, label_b = self.cal_loss(output_b, inputs['label'])
        
        un_loss_a, un_accuracy_a, un_pred_label_id_a, un_label_a = self.cal_loss(un_output_a, un_gt_a)
        un_loss_b, un_accuracy_b, un_pred_label_id_b, un_label_b = self.cal_loss(un_output_b, un_gt_b)

        return loss_a, un_loss_a, loss_b, un_loss_b, accuracy_a, accuracy_b
    
    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
        

def init_weights_b(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
        