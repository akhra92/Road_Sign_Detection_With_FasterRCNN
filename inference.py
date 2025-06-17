from visualize import tensor_2_im
import random
import matplotlib.pyplot as plt
import torch
import cv2
import config as cfg
from dataloader import get_dls
from visualize import get_tfs
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def inference(model, ts_dl, num_ims, rows, class_names, threshold = 0.3, cmap = None):
    
    assert cmap in ["rgb", "gray"], "Rasmni oq-qora yoki rangli ekanini aniqlashtirib bering!"
    if cmap == "rgb": cmap = "viridis"
    
    plt.figure(figsize = (20, 10))
    indekslar = [random.randint(0, len(ts_dl) - 1) for _ in range(num_ims)]
    
    for idx, indeks in enumerate(indekslar):
        im, _ = ts_dl.dataset[indeks]
        with torch.no_grad(): predictions = model(im.unsqueeze(0).to(cfg.DEVICE))
        img = tensor_2_im(im, inv_trans = True)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bbox_class_names = []
        for i, (boxes, scores, labels) in enumerate(zip(predictions[0]["boxes"], predictions[0]["scores"], predictions[0]["labels"])):
            if scores > threshold:
                bbox_class_names.append(list(class_names.keys())[labels.item()])
                r, g, b = [random.randint(0, 255) for _ in range(3)]
                x, y, w, h = [round(b.item()) for b in boxes]
                cv2.rectangle(img = img, pt1 = (x, y), pt2 = (w, h), color = (r, g, b), thickness = 2)
        plt.subplot(rows, num_ims // rows, idx + 1)
        plt.imshow(img); plt.title(f"{bbox_class_names}")
        plt.savefig('figure1.png')
        plt.axis("off")

if __name__ == '__main__':
    tfs = get_tfs()
    tr_dl, val_dl, ts_dl, class_names = get_dls(root=cfg.DATA_PATH, transforms=tfs, bs=cfg.BATCH_SIZE)
    m = torch.load('saved_models/road_best_model.pt', weights_only=False).to(cfg.DEVICE)
    inference(model = m, ts_dl = ts_dl, num_ims = 12, rows = 3, cmap = "rgb", class_names=class_names)