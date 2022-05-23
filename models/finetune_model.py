import torch
import torch.nn as nn
import torch.nn.functional as F
from .lxrt import LXRTModel, GeLU, BertLayerNorm

class ClassificationModel(nn.Module):

    def __init__(self, num_class):
        super().__init__()
        self.bert = LXRTModel.from_pretrained("bert-base-uncased")
        hid_dim = 768

        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_class)
        )
        self.logit_fc.apply(self.bert.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                visual_feats=None, visual_attention_mask=None,
                category_label=None, inference=False):
        _, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask,
            visual_feats=visual_feats, visual_attention_mask=visual_attention_mask
        )
        logit = self.logit_fc(pooled_output) # [B, cat_len]

        if inference:
            return torch.argmax(logit, dim=1)
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