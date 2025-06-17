import xmltodict
import torch
from glob import glob
from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader


class RoadSignDetection(Dataset):

    def __init__(self, root, transforms = None, size = 224):
        self.root = root
        self.size = size
        self.transforms = transforms
        self.class_names = {}
        self.class_counts = 0
        self.data_info = sorted(glob(f'{root}/annotations/*.xml'))

        for idx, data_info in enumerate(self.data_info):
            info = xmltodict.parse(open(f'{data_info}', 'r').read())
            bbox_info = info['annotation']['object']
            if isinstance(bbox_info, list):
                for bb_info in bbox_info:                    
                    if class_name not in self.class_names:
                        self.class_names[class_name] = self.class_counts
                        self.class_counts += 1
            elif isinstance(bbox_info, dict):
                for bb_info in bbox_info:
                    class_name = bbox_info['name']
                    if class_name not in self.class_names:
                        self.class_names[class_name] = self.class_counts
                        self.class_counts += 1

    def __len__(self):
        return len(self.data_info)
    
    def get_info(self, idx):
        return xmltodict.parse(open(f'{self.data_info[idx]}', 'r').read())
    
    def read_img(self, info):
        return Image.open(f'{self.root}/images/{info["annotation"]["filename"]}').convert('RGB')
    
    def get_coordinates(self, bbox):
        return float(bbox['xmin']), float(bbox['ymin']), float(bbox['xmax']), float(bbox['ymax'])
    
    def get_bboxes(self, info):
        bboxes, class_names = [], []
        bbox_info = info['annotation']['object']

        if isinstance(bbox_info, list):
            for bb_info in bbox_info:
                bbox = bb_info['bndbox']
                class_name = bb_info['name']
                class_names.append(class_name)
                bboxes.append(self.get_coordinates(bbox))

        elif isinstance(bbox_info, dict):
            for bb_info in bbox_info:
                if bb_info == 'bndbox':
                    bbox = bbox_info[bb_info]
                    class_name = bbox_info['name']
                    bboxes.append(self.get_coordinates(bbox))
                    class_names.append(class_name)

        return torch.as_tensor(bboxes, dtype=torch.float32), class_names
    
    def get_label(self, class_name):
        return self.class_names[class_name]
    
    def get_area(self, bboxes):
        return (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
    
    def create_target(self, bboxes, labels, img_id, area, is_crowd):        
        target = {}
        target['boxes'] = bboxes
        target['labels'] = labels
        target['image_id'] = img_id
        target['area'] = area
        target['iscrowd'] = is_crowd

        return target
    
    def __getitem__(self, idx):        
        info = self.get_info(idx)
        image = self.read_img(info)
        bboxes, class_names = self.get_bboxes(info)
        labels = torch.tensor([self.get_label(class_name) for class_name in class_names], dtype=torch.int64)
        img_id = torch.tensor([idx])
        area = self.get_area(bboxes)
        is_crowd = torch.zeros((len(bboxes), ), dtype=torch.int64)
        target = self.create_target(bboxes, labels, img_id, area, is_crowd)

        if self.transforms:
            image = self.transforms(image)

        return image, target
    

def custom_collate_fn(batch):
    return tuple(zip(*batch))

def get_dls(root, transforms, bs, split=[0.8, 0.1], ns=4):    
    dataset = RoadSignDetection(root, transforms)

    all_len = len(dataset)
    tr_len = int(all_len*split[0])
    val_len = int(all_len*split[1])

    trn_ds, val_ds, test_ds = random_split(dataset, lengths=[tr_len, val_len, all_len-tr_len-val_len])

    trn_dl = DataLoader(trn_ds, batch_size=bs, collate_fn=custom_collate_fn, shuffle=True, num_workers=ns)
    val_dl = DataLoader(val_ds, batch_size=bs, collate_fn=custom_collate_fn, shuffle=False, num_workers=ns)
    test_dl = DataLoader(test_ds, batch_size=1, collate_fn=custom_collate_fn, shuffle=False, num_workers=ns)

    return trn_dl, val_dl, test_dl, dataset.class_names