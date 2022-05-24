import json
import random
import zipfile
from io import BytesIO
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer


def create_dataloaders(args):
    dataset = LXRTDataset(args=args,
                          ann_paths=[args.labeled_annotation, args.unlabeled_annotation],
                          labeled_zip_feats=args.labeled_zip_feats,
                          unlabeled_zip_feats=args.unlabeled_zip_feats)
    size = len(dataset)
    val_size = int(size * args.val_ratio)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size - val_size, val_size],
                                                               generator=torch.Generator().manual_seed(args.seed))

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers)
                                  #prefetch_factor=args.prefetch)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.val_batch_size,
                                sampler=val_sampler,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=args.num_workers)
                                #prefetch_factor=args.prefetch)
    return train_dataloader, val_dataloader


class LXRTDataset(Dataset):
    """ A simple class that supports multi-modal inputs.

    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the title information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.

    Args:
        ann_paths (list): list of annotation file path, with the '.json' suffix.
        labeled_zip_feat (str): labeled visual feature zip file path.
        unlabeled_zip_feat(str): unlabeled visual feature zip file path
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 ann_paths: List[str],
                 labeled_zip_feats: str,
                 unlabeled_zip_feats: str):
        self.max_frame = args.max_frames # 32
        self.bert_seq_length = args.bert_seq_length # 50

        # lazy initialization for zip_handler to avoid multiprocessing-reading error
        self.labeled_zip_feat_path = labeled_zip_feats
        self.unlabeled_zip_feat_path = unlabeled_zip_feats
        self.handles = [{'labeled':None, 'unlabeled':None} for _ in range(args.num_workers)]

        # load annotations
        self.anns = []
        for ann_path in ann_paths:
            with open(ann_path, 'r', encoding='utf8') as f:
                self.anns.extend(json.load(f))

        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

    def __len__(self) -> int:
        return len(self.anns)

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        worker_info = torch.utils.data.get_worker_info()
        frame_input, frame_mask = self.get_visual_feats(worker_info.id, idx) # [max_frame, feat_dim], [max_frame]
        masked_frame_input, origin_frame_feat, frame_mask_label, frame_mask = self.proc_visual_feats(worker_info.id, frame_input, frame_mask)

        annotation = self.anns[idx]
        is_matched = 1
        if random.random() < 0.5:
            annotation = self.anns[random.randint(0, len(self.anns) - 1)]
            is_matched = 0

        sentence = annotation['title'] + "[SEP]" + annotation['asr']
        for ocr in annotation['ocr']:
            sentence = sentence + "[SEP]" + ocr['text']

        masked_input_ids, mlm_labels, mask = self.proc_text(sentence)
        is_matched = torch.tensor(is_matched, dtype=torch.long)

        data = dict(
            frame_input=masked_frame_input,
            origin_frame=origin_frame_feat,
            frame_mask_label=frame_mask_label,
            frame_mask=frame_mask,
            title_input=masked_input_ids,
            title_mlm_label=mlm_labels,
            title_mask=mask,
            is_matched=is_matched
        )

        return data

    def get_visual_feats(self, worker_id, idx: int) -> tuple:
        # read data from zipfile
        annotation = self.anns[idx]
        vid = annotation['id']
        if "category_id" in annotation:
            data_type = 'labeled'
            zip_feat_path = self.labeled_zip_feat_path
        else:
            data_type = 'unlabeled'
            zip_feat_path = self.unlabeled_zip_feat_path

        if self.handles[worker_id][data_type] is None:
            self.handles[worker_id][data_type] = zipfile.ZipFile(zip_feat_path, 'r')
        raw_feats = np.load(BytesIO(self.handles[worker_id][data_type].read(name=f'{vid}.npy')), allow_pickle=True)
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample the frames.
            # randomly sample when test mode is False
            select_inds = list(range(num_frames))
            random.shuffle(select_inds)
            select_inds = select_inds[:self.max_frame]
            select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]

        return feat, mask

    def proc_text(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        # encoded_inputs:
        # {'input_ids':[], 'token_type_ids':[], 'attention_mask':[]}
        masked_input_ids, mlm_labels = self.random_word(encoded_inputs['input_ids'])

        masked_input_ids = torch.LongTensor(masked_input_ids) # [101, ..., 102, 0, 0, ..., 0]
        mlm_labels = torch.LongTensor(mlm_labels)
        mask = torch.LongTensor(encoded_inputs['attention_mask']) # [1,1,..,1,0,0,..,0]

        assert masked_input_ids.size(0) == self.bert_seq_length
        assert mlm_labels.size(0) == self.bert_seq_length
        assert mask.size(0) == self.bert_seq_length

        return masked_input_ids, mlm_labels, mask

    def proc_visual_feats(self, worker_id, feat, mask) -> tuple:
        mask_feat, feat_mask_label = self.random_feat(worker_id, feat)

        mask_feat = torch.FloatTensor(mask_feat)
        origin_feat = torch.FloatTensor(feat)
        feat_mask_label = torch.LongTensor(feat_mask_label)
        mask = torch.LongTensor(mask)

        assert mask_feat.size(0) == self.max_frame
        assert origin_feat.size(0) == self.max_frame
        assert feat_mask_label.size(0) == self.max_frame
        assert mask.size(0) == self.max_frame

        return mask_feat, origin_feat, feat_mask_label, mask

    def random_word(self, token_ids):
        '''
        :param token_ids: input of token ids
        :return: (list of tokens_ids(some substituted by [MASK]), list of int where unmasked place is -1)
        '''
        mlm_label = []
        for i, token_id in enumerate(token_ids):
            if token_id == 0 or token_id == 101 or token_id == 102:
                mlm_label.append(-1)

            else:
                prob = random.random()
                if prob < 0.15: # 15%
                    prob /= 0.15
                    if prob < 0.8: # mask
                        token_ids[i] = self.tokenizer.vocab["[MASK]"]
                    elif prob < 0.9: # random replace
                        token_ids[i] = random.choice(list(self.tokenizer.vocab.values()))

                    try:
                        mlm_label.append(token_id)
                    except KeyError:
                        mlm_label.append(self.tokenizer.vocab["[UNK]"])
                else:
                    mlm_label.append(-1)

        return token_ids, mlm_label

    def random_feat(self, worker_id, feats):
        '''
        :param feats: numpy array [max_frame, feat_dim]
        :return: (masked image features, 1-D numpy array where masked position is 1 otherwise 0)
        '''
        mask_feats = feats.copy()
        feat_mask_label = np.zeros(len(feats), dtype=np.float32)
        for i in range(len(feats)):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8: # mask
                    mask_feats[i, :] = 0.
                elif prob < 0.9: # random replace
                    # substitute with a random feat
                    rand_idx = random.randint(0, len(self.anns) - 1)
                    rand_feats, _ = self.get_visual_feats(worker_id, rand_idx)
                    mask_feats[i, :] = rand_feats[random.randint(0,31)]

                feat_mask_label[i] = 1.

        return mask_feats, feat_mask_label
