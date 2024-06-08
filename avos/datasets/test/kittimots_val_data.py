"""
kittimots dataloader for inference.
"""
import math
import torch
import torch.utils.data
import os
from PIL import Image
import cv2
import numpy as np
import glob
import logging
import json

from avos.datasets import path_config as dataset_path_config
import avos.datasets.transforms as T
# import avos.datasets.transforms_kitti as T

# # Resize to specific width and height
# resize_transform = T.Resize((1242, 375))
#
# # Resize while keeping aspect ratio, by specifying shorter side
# resize_transform = T.Resize(256)  # Resize shorter side to 256 pixels

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class KittimotsValDataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=6, val_size=473, sequence_names=None, use_flow=False, max_sc=None, style=None):
        super(KittimotsValDataset, self).__init__()
        print('KittimotsValDataset->use_flow:'+str(use_flow))
        self.num_frames = num_frames
        self.split = 'val'
        self.use_flow = use_flow
        self.im_size = val_size
        self.style = style
        # self._transforms = make_validation_transforms(min_size=val_size, max_sc=max_sc) #davis
        self._transforms = make_validation_transforms(min_size=val_size, max_sc=max_sc) #kittimots

        self.kittimots_val_json_file = dataset_path_config.kittimots_val_json_file
        self.kittimots_rgb_training_path = dataset_path_config.kittimots_rgb_training_path
        self.kittimots_gt_path = dataset_path_config.kittimots_gt_path
        self.kittimots_flow_path = dataset_path_config.kittimots_flow_path
        self.all_styles= dataset_path_config.kittimots_styles
        self.stylized_kittimots_path = dataset_path_config.kittimots_stylized_image_path

        if style is None or style == 'None':
            style = 'Original'
        print(style)
        if style != 'Original':
            assert style in self.all_styles

        self.frames_info = {
            'kittimots': {}
        }
        self.img_ids = []
        logger.debug('loading kittimots val json file...')
        with open(self.kittimots_val_json_file, 'r') as f:
            data = json.load(f)
            # Extract filenames from the "images" list
            filename_array = [image_info['file_name'][:4] for image_info in data['images'] if
                              image_info['file_name'].split('/')[-1].split('.')[0] == '000055']

            for video_name in filename_array:
                # Join the file path and image name
                frames = sorted(glob.glob(os.path.join(self.kittimots_rgb_training_path, video_name, '*.png')))
                self.frames_info['kittimots'][video_name] = [frame_path.split('/')[-1][:-4] for frame_path in frames]
                self.img_ids.extend([('kittimots', video_name, frame_index) for frame_index in range(len(frames))])
                # print(self.frames_info['kittimots'][video_name])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_ids_i = self.img_ids[idx]
        dataset, video_name, frame_index = img_ids_i
        vid_len = len(self.frames_info[dataset][video_name])
        center_frame_name = self.frames_info[dataset][video_name][frame_index]
        frame_indices = [(x + vid_len) % vid_len for x in range(frame_index - math.floor(float(self.num_frames) / 2),
                                                                frame_index + math.ceil(float(self.num_frames) / 2), 1)]
        assert len(frame_indices) == self.num_frames
        frame_ids = []
        img = []
        flows = []
        masks = []
        mask_paths = []
        skip_current_sample_flow = False
        img_paths = []
        for frame_id in frame_indices:
            frame_name = self.frames_info[dataset][video_name][frame_id]
            frame_ids.append(frame_name)
            if self.style is None or self.style == 'Original':
                img_path = os.path.join(self.kittimots_rgb_training_path, video_name, frame_name + '.png')
            else:
                img_path = os.path.join(self.stylized_kittimots_path, 'JPEGImages', self.style, video_name,
                                        frame_name + '.jpg')
            # img_path = os.path.join(self.kittimots_rgb_training_path, video_name, frame_name + '.jpg')
            gt_path = os.path.join(self.kittimots_gt_path, video_name, frame_name + '.png')
            img_i = Image.open(img_path).convert('RGB')
            img.append(img_i)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt[gt > 0] = 255
            masks.append(torch.Tensor(np.expand_dims(np.asarray(gt.copy()), axis=0)))
            mask_paths.append(gt_path)
            img_paths.append(img_path)
            if self.use_flow and not skip_current_sample_flow:
                prev_frame = '%05d' % (int(frame_name) - 1)
                next_frame = '%05d' % (int(frame_name) + 1)
                fwd_flow_file = os.path.join(self.kittimots_flow_path, video_name,
                                             prev_frame + '_' + frame_name + '.png')
                bkd_flow_file = os.path.join(self.kittimots_flow_path, video_name,
                                             frame_name + '_' + next_frame + '.png')
                fwd_flow = None
                bkd_flow = None
                if os.path.exists(fwd_flow_file):
                    fwd_flow = Image.open(fwd_flow_file).convert('RGB')
                elif os.path.exists(bkd_flow_file):
                    bkd_flow = Image.open(bkd_flow_file).convert('RGB')
                if fwd_flow is not None:
                    flow = fwd_flow
                elif bkd_flow is not None:
                    flow = bkd_flow
                else:
                    skip_current_sample_flow = True
                    logger.debug('Flow not found for fwd_flow_file: : ' + fwd_flow_file)
                    logger.debug('Flow not found for bkd_flow_file: ' + bkd_flow_file)
                    flow = None
                flows.append(flow)
        masks = torch.cat(masks, dim=0)
        target = {'dataset': dataset, 'video_name': video_name, 'center_frame': center_frame_name,
                  'frame_ids': frame_ids, 'masks': masks, 'vid_len': vid_len, 'mask_paths': mask_paths,
                  'img_paths': img_paths}
        if self.use_flow and not skip_current_sample_flow:
            target['flows'] = flows
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        """        
        if self.use_flow and not skip_current_sample_flow:
            target['flows'] = []
            for i in range(len(img)):
                if target['flows'][i].shape[-2:] != img[i].shape[-2:]:
                    target['flows'][i] = torch.nn.functional.interpolate(target['flows'][i].unsqueeze(0),
                                                                         img[i].shape[-2:])
                else:
                    target['flows'][i] = target['flows'][i].unsqueeze(0)
            target['flows'] = torch.cat(target['flows'], dim=0)
        """
        if self.use_flow and not skip_current_sample_flow:
            target['flows'] = [torch.nn.functional.interpolate(target['flows'][i].unsqueeze(0), img[i].shape[-2:]) if target['flows'][i].shape[-2:] !=  img[i].shape[-2:] else target['flows'][i].unsqueeze(0) for i in range(len(img))]
            target['flows'] = torch.cat(target['flows'], dim=0)
        return torch.cat(img, dim=0), target


def make_validation_transforms(min_size=360, max_sc=None):
    if max_sc is None:
        max_sc = 1.8
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    ])
    return T.Compose([
        T.RandomResize([min_size], max_size=int(max_sc * min_size)),
        normalize,
    ])




