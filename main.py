from train import train_one_epoch, evaluate
import os
import torch
import config as cfg
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from dataloader import get_dls
import torchvision.transforms as T


def get_tfs():
    return T.Compose([T.ToTensor(), T.ConvertImageDtype(dtype=torch.float), T.Normalize(mean=cfg.MEAN, std=cfg.STD)])


def run():
    m = fasterrcnn_resnet50_fpn(weights = 'DEFAULT')    
    in_features = m.roi_heads.box_predictor.cls_score.in_features    
    m.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, cfg.NUM_CLASSES)
    m.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, cfg.NUM_CLASSES * 4)
    m.to(cfg.DEVICE)

    tfs = get_tfs()
    tr_dl, val_dl, ts_dl, class_names = get_dls(root=cfg.DATA_PATH, transforms=tfs, bs=cfg.BATCH_SIZE)
    optimizer = torch.optim.SGD(m.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(cfg.EPOCHS):        
        train_one_epoch(m, optimizer, tr_dl, device=cfg.DEVICE, epoch=epoch, print_freq = 10)        
        lr_scheduler.step()        
        evaluate(m, val_dl, device = cfg.DEVICE)

    os.makedirs(f"{cfg.SAVE_MODEL_PATH}", exist_ok = True)
    torch.save(m, f"{cfg.SAVE_MODEL_PATH}/{cfg.MODEL_PREFIX}_best_model.pt")
    print("Train has finished!")


if __name__ == '__main__':
    run()