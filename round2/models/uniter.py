import sys
sys.path.append('../')
from config import args
import numpy as np
import torch
from torch import nn
import logging

from transformers.models.bert.modeling_bert import BertConfig, BertPreTrainedModel, BertEmbeddings, \
    BertEncoder, BertOnlyMLMHead, BertPredictionHeadTransform


class UniModel(BertPreTrainedModel):

    def __init__(self, config, img_dim):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.video_fc = torch.nn.Linear(img_dim, config.hidden_size)
        
        self.encoder = BertEncoder(config)

        self.init_weights()

    def forward(self, input_ids, text_attention_mask, video_feat=None, video_attention_mask=None):
        text_emb = self.embeddings(input_ids=input_ids)
        if video_feat is not None:
            video_emb = self.video_fc(video_feat)
            embedding_output = torch.cat([text_emb, video_emb], dim=1)
            attention_mask = torch.cat([text_attention_mask, video_attention_mask], dim=1)
        else:
            embedding_output = text_emb
            attention_mask = text_attention_mask

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoded_layers = self.encoder(embedding_output, extended_attention_mask)['last_hidden_state']

        return encoded_layers


class UniterPretraining(nn.Module):
    def __init__(self,
                 task_mask_lm=True,
                 task_matched=True,
                 task_mrfr=True):
        super().__init__()
        # Configuration
        config = BertConfig.from_pretrained(args.bert_dir)
        self.config = config

        # Use of pre-training tasks
        self.task_mask_lm = task_mask_lm
        self.task_mrfr = task_mrfr
        self.task_matched = task_matched

        self.roberta = UniModel.from_pretrained(args.bert_dir, img_dim=args.frame_embedding_size)

        # Pre-training heads
        self.cls = BertOnlyMLMHead(config)
        if self.task_mrfr:
            self.obj_predict_head = VisualOnlyMLMHead(config)

        if self.task_matched:
            self.itm_head = nn.Linear(config.hidden_size, 1)

    def _compute_masked_hidden(self, hidden_states, mask):
        """ get only the masked region (don't compute unnecessary hiddens) \
            refer to https://github.com/ChenRocks/UNITER/blob/master/model/pretrain.py"""
        # mask is of size [b, txt_len] with elements True/False
        mask = mask.unsqueeze(-1).expand_as(hidden_states)
        hidden_masked = hidden_states[mask].contiguous().view(-1, hidden_states.size(-1))
        return hidden_masked

    def forward(self, title_ids, title_mask=None, masked_lm_labels=None,
                visual_feats=None, visual_mask=None, origin_feat=None, mfm_label=None, matched_label=None):
        
        sequence_output = self.roberta(title_ids, title_mask, visual_feats, visual_mask)
        assert sequence_output.size(1) == args.bert_seq_length + args.max_frames
        title_output, visn_output = torch.split(sequence_output, [args.bert_seq_length, args.max_frames], dim=1)

        title_output_masked = self._compute_masked_hidden(title_output, masked_lm_labels != -1)
        lang_prediction_scores = self.cls(title_output_masked)

        total_loss = 0.
        losses = ()
        if masked_lm_labels is not None and self.task_mask_lm:
            masked_lm_loss = nn.CrossEntropyLoss()(lang_prediction_scores, masked_lm_labels[masked_lm_labels != -1])
            total_loss += masked_lm_loss
            losses += (masked_lm_loss.detach(),)

        if matched_label is not None and self.task_matched:
            match_pred = self.itm_head(sequence_output[:, 0, :]) # [CLS] representation
            matched_loss = nn.BCEWithLogitsLoss()(match_pred.view(-1), matched_label.view(-1))

            total_loss += matched_loss
            losses += (matched_loss.detach(),)

        if origin_feat is not None and self.task_mrfr:
            visn_output_masked = self._compute_masked_hidden(visn_output, mfm_label == 1)
            pred_visual_feat = self.obj_predict_head(visn_output_masked)
            origin_feat_masked = self._compute_masked_hidden(origin_feat, mfm_label == 1)
            visn_loss = nn.SmoothL1Loss()(pred_visual_feat, origin_feat_masked)

            total_loss += visn_loss
            losses += (visn_loss.detach(),)

        return total_loss, torch.stack(losses).unsqueeze(0)


class VisualLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, args.frame_embedding_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(args.frame_embedding_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


