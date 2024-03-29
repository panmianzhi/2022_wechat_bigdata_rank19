import os
import io
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import zipfile
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor
import clip

class RawFrameDataset(Dataset):

    def __init__(self,
                 ann_path: str,
                 zip_frame_dir: str,
                 max_video_frames: int = 32):
        """ This class is used to load raw video frames.
        Args:
            ann_paths (str): the annotation file path.
            zip_frame_dir (str): the directory that saves zip frames.
            max_video_frames (str): the maximum number of video frames.
        """
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        self.zip_frame_dir = zip_frame_dir
        self.max_video_frames = max_video_frames
        
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)) # clip
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 官方版本
        # we follow the common practice as in the ImageNet's preprocessing.
        self.transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
    def __len__(self) -> dict:
        return len(self.anns)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ Extract the frame tensor from zipped file.
        The output tensor is in shape of [MAX_FRAMES, 3, 224, 224]
        """
        feedid = self.anns[idx]['id']
        zip_path = os.path.join(self.zip_frame_dir, feedid[-3:], f'{feedid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        img_name_list = handler.namelist()
        img_name_list = sorted(img_name_list)
        img_name_list = img_name_list[:self.max_video_frames]
        img_tensor = torch.zeros(self.max_video_frames, 3, 224, 224)
        for i, img_name in enumerate(img_name_list):
            i_img_content = handler.read(img_name)
            i_img = Image.open(io.BytesIO(i_img_content))
            i_img_tensor = self.transform(i_img)
            img_tensor[i, ...] = i_img_tensor
        handler.close()
        num_frames = torch.LongTensor([len(img_name_list)])
        return dict(img=img_tensor, num_frames=num_frames)


def parse_args():
    parser = argparse.ArgumentParser("Visual feature extraction")
    parser.add_argument('--zip_frame_dir', type=str, default='/home/tione/notebook/data/zip_frames/unlabeled_new/')
    parser.add_argument('--ann_path', type=str, default='/home/tione/notebook/data/annotations/unlabeled_new.json')
    parser.add_argument('--output_path', type=str, default='./clip_features/unlabeled.zip')
    parser.add_argument('--clip_encoder', type=str, default='./clip_ckpt_file/ViT-B-32.pt')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_clip_encoder, preprocess = clip.load(args.clip_encoder, device)
    print("clip model load done")
    
    dataset = RawFrameDataset(args.ann_path, args.zip_frame_dir)
    # batch-size == 8 is fine for V100 GPU, please consider use smaller batch-size if OOM issue occurs.
    dataloader = DataLoader(dataset, batch_size=8, num_workers=16, shuffle=False, pin_memory=True, drop_last=False)

    assert not os.path.isfile(args.output_path), f"{args.output_path} already exists. " \
                                                 "If you want to override it, please manually delete this file."
    output_handler = zipfile.ZipFile(args.output_path, 'w', compression=zipfile.ZIP_STORED)

    with torch.no_grad():
        cur = 0
        for dataitem in dataloader:
            img, num_frames = dataitem['img'], dataitem['num_frames']
            B, L = img.shape[0:2]
            img = img.view((B * L,) + img.shape[2:])
            feature = model_clip_encoder.encode_image(img.to(device))
            feature = feature.view(B, L, -1)
            feature = feature.cpu().numpy().astype(np.float32) #?
            for i in range(B):
                feedid = dataset.anns[cur]['id']
                ioproxy = io.BytesIO()
                np.save(ioproxy, feature[i, :int(num_frames[i])])
                npy_str = ioproxy.getvalue()
                output_handler.writestr(f'{feedid}.npy', npy_str)
                cur += 1
                if cur % 1000 == 0:
                    print(f"Extract feature {cur}/{len(dataset)}, feat size {feature.shape}")
    output_handler.close()


if __name__ == '__main__':
    main()