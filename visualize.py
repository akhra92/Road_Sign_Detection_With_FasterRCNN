import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
import config as cfg
from dataloader import get_dls
import torch


def tensor_2_im(t, t_type = "rgb", inv_trans = False):
    
    assert t_type in ["rgb", "gray"], "Define if an image is RGB or Grayscale!"
    gray_tfs = T.Compose([T.Normalize(mean = [ 0.], std = [1/0.5]), T.Normalize(mean = [-0.5], std = [1])])
    rgb_tfs  = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]), T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])
    
    invTrans = gray_tfs if t_type == "gray" else rgb_tfs 
    
    return (invTrans(t) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8) if inv_trans else (t * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)


def get_tfs():
    return T.Compose([T.ToTensor(), T.ConvertImageDtype(dtype=torch.float), T.Normalize(mean=cfg.MEAN, std=cfg.STD)])


def visualize(data, n_ims, rows, class_names, cmap = None):
    
    assert cmap in ["rgb", "gray"], "Define if an image is RGB or Grayscale!"
    if cmap == "rgb": cmap = "viridis"
    
    plt.figure(figsize = (20, 10))
    indekslar = [random.randint(0, len(data) - 1) for _ in range(n_ims)]
    for idx, indeks in enumerate(indekslar):
        im, target = data[indeks]
        img = tensor_2_im(im, inv_trans = True)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bbox_class_names = []
        for i, cntr in enumerate(target["boxes"]):
            r, g, b = [random.randint(0, 255) for _ in range(3)]
            x, y, w, h = [round(c.item()) for c in cntr]
            bbox_class_names.append(list(class_names.keys())[target["labels"][i].item()])
            cv2.rectangle(img = img, pt1 = (x, y), pt2 = (w, h), color = (r, g, b), thickness = 2)
        bbox_class_names = [bbox_class_name for bbox_class_name in bbox_class_names]
        plt.subplot(rows, n_ims // rows, idx + 1)        
        plt.imshow(img); plt.title(f"{bbox_class_names}")
        plt.savefig('figure.png')
        plt.axis("off")

if __name__ == '__main__':
    tfs = get_tfs()
    tr_dl, val_dl, ts_dl, class_names = get_dls(root=cfg.DATA_PATH, transforms=tfs, bs=cfg.BATCH_SIZE)    
    visualize(tr_dl.dataset, 20, 4, class_names, "rgb")