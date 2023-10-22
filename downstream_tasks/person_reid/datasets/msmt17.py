import os
import re
import glob
from PIL import Image

from torch.utils.data import Dataset


class MSMT17(Dataset):
    """Build MSMT17 dataset.
    
    Data directory structure:
    - data_path/
        - bounding_box_train/
        - bounding_box_test/
        - query/
    """
    def __init__(self, cfg, transform, is_train):
        self.transform = transform

        if is_train:
            self.train_dir = os.path.join(cfg.DATA.ROOT_DIR, 'bounding_box_train')
            self.img_items = self.process_dir(self.train_dir)
        else:
            self.query_dir = os.path.join(cfg.DATA.ROOT_DIR, 'query')
            query_img_items = self.process_dir(self.query_dir)
            self.gallery_dir = os.path.join(cfg.DATA.ROOT_DIR, 'bounding_box_test')
            gallery_img_items = self.process_dir(self.gallery_dir)
            self.img_items = query_img_items + gallery_img_items
            self.num_queries = len(query_img_items)

        pid_set = set()
        cam_set = set()

        for img_item in self.img_items:
            pid_set.add(img_item[1])
            cam_set.add(img_item[2])
        
        pids = sorted(list(pid_set))
        cams = sorted(list(cam_set))
        self.pid_dict = dict([(p, i) for i, p in enumerate(pids)])
        self.cam_dict = dict([(p, i) for i, p in enumerate(cams)])
        self.num_classes = len(pids)
    
    def process_dir(self, dir_path):
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 1 <= camid <= 15
            camid -= 1  # index starts from 0
            data.append((img_path, pid, camid))

        return data

    def __len__(self):
        return len(self.img_items)
    
    def __getitem__(self, i):
        img_path, pid, camid = self.img_items[i]

        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.transform(img)
        pid = self.pid_dict[pid]
        camid = self.cam_dict[camid]

        return (img, pid, camid)