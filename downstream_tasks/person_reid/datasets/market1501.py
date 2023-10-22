import os
import re
import glob

from PIL import Image
from torch.utils.data import Dataset


class Market1501(Dataset):
    """Build Market-1501 dataset.

    Data directory structure:
    - data_path/
        - bounding_box_train/
        - bounding_box_test/
        - query/
    """
    def __init__(self, cfg, transform, is_train):
        self.transform = transform
        self.num_classes = None
        self.num_queries = None

        train_dir = os.path.join(cfg.DATA.ROOT_DIR, "bounding_box_train")
        query_dir = os.path.join(cfg.DATA.ROOT_DIR, 'query')
        gallery_dir = os.path.join(cfg.DATA.ROOT_DIR, 'bounding_box_test')

        if is_train:
            self.img_items = self.process_dir(train_dir)
        else:
            query_img_items = self.process_dir(query_dir)
            gallery_img_items = self.process_dir(gallery_dir)
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

    def process_dir(self, root_dir):
        img_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            data.append((img_path, pid, camid))
        
        return data
    
    def __len__(self):
        return len(self.img_items)
    
    def __getitem__(self, i):
        img_item = self.img_items[i]
        img_path, pid, camid = img_item

        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.transform(img)
        pid = self.pid_dict[pid]
        camid = self.cam_dict[camid]

        return (img, pid, camid)
