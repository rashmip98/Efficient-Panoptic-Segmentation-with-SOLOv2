from datasets.panoptic_dataset import PanopticDataset, collate_fn
from efficientps import EffificientPS
import albumentations as A
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from detectron2.config import get_cfg, CfgNode
from detectron2.utils.events import _CURRENT_STORAGE_STACK, EventStorage
import os
import torch
import numpy as np
from panopticapi.evaluation import pq_compute
from tqdm import tqdm
from efficientps.panoptic_metrics import generate_pred_panoptic
from efficientps.panoptic_segmentation_module import panoptic_segmentation_module
from panopticapi.utils import id2rgb
import torch.nn.functional as F
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

def add_custom_param(cfg):
    """
    In order to add custom config parameter in the .yaml those parameter must
    be initialised
    """
    # Model
    cfg.MODEL_CUSTOM = CfgNode()
    cfg.MODEL_CUSTOM.BACKBONE = CfgNode()
    cfg.MODEL_CUSTOM.BACKBONE.EFFICIENTNET_ID = 5
    cfg.MODEL_CUSTOM.BACKBONE.LOAD_PRETRAIN = False
    # DATASET
    cfg.NUM_CLASS = 19
    cfg.DATASET_PATH = "/home/ubuntu/Elix/cityscapes"
    cfg.TRAIN_JSON = "gtFine/cityscapes_panoptic_train.json"
    cfg.VALID_JSON = "gtFine/cityscapes_panoptic_val.json"
    cfg.PRED_DIR = "preds"
    cfg.PRED_JSON = "cityscapes_panoptic_preds.json"
    # Transfom
    cfg.TRANSFORM = CfgNode()
    cfg.TRANSFORM.NORMALIZE = CfgNode()
    cfg.TRANSFORM.NORMALIZE.MEAN = (106.433, 116.617, 119.559)
    cfg.TRANSFORM.NORMALIZE.STD = (65.496, 67.6, 74.123)
    cfg.TRANSFORM.RESIZE = CfgNode()
    cfg.TRANSFORM.RESIZE.HEIGHT = 512
    cfg.TRANSFORM.RESIZE.WIDTH = 1024
    cfg.TRANSFORM.RANDOMCROP = CfgNode()
    cfg.TRANSFORM.RANDOMCROP.HEIGHT = 512
    cfg.TRANSFORM.RANDOMCROP.WIDTH = 1024
    cfg.TRANSFORM.HFLIP = CfgNode()
    cfg.TRANSFORM.HFLIP.PROB = 0.5
    # Solver
    cfg.SOLVER.NAME = "SGD"
    cfg.SOLVER.ACCUMULATE_GRAD = 1
    # Runner
    cfg.BATCH_SIZE = 1
    cfg.CHECKPOINT_PATH = ""
    cfg.PRECISION = 32
    # Callbacks
    cfg.CALLBACKS = CfgNode()
    cfg.CALLBACKS.CHECKPOINT_DIR = None
    # Inference
    cfg.INFERENCE = CfgNode()
    cfg.INFERENCE.AREA_TRESH = 0

def main():

  cfg = get_cfg()
  add_custom_param(cfg)
  cfg.merge_from_file("config.yaml")
  score_thr = 0.3

  transform_valid = A.Compose([
        A.Resize(height=512, width=1024),
        A.Normalize(mean=cfg.TRANSFORM.NORMALIZE.MEAN,
                    std=cfg.TRANSFORM.NORMALIZE.STD),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
  
  inv_normalize = torchvision.transforms.Normalize(
    mean=[-106.433/65.496, -116.617/67.6, -119.559/74.123],
    std=[1/65.496, 1/67.6, 1/74.123]
      )

  valid_dataset = PanopticDataset(cfg.VALID_JSON,
                                    cfg.DATASET_PATH,
                                    'val',
                                    transform=transform_valid)
  valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=False,
        num_workers=4
    )        
  if os.path.exists(cfg.CHECKPOINT_PATH):
      print('""""""""""""""""""""""""""""""""""""""""""""""')
      print("Loading model from {}".format(cfg.CHECKPOINT_PATH))
      print('""""""""""""""""""""""""""""""""""""""""""""""')
      efficientps = EffificientPS.load_from_checkpoint(cfg=cfg,
          checkpoint_path=cfg.CHECKPOINT_PATH)
  np_paths = os.listdir('/content/drive/MyDrive/EfficientPS/solo_outputs')
  
  result_l = []
  efficientps.eval().cuda()
  with torch.no_grad():
      for i, data in enumerate(tqdm(valid_loader)):
        # Load the mask outputs of SOLO
        solo_mask = np.load('/content/drive/MyDrive/EfficientPS/solo_outputs/'+np_paths[i], allow_pickle=True)

        solo_mask = torch.Tensor(solo_mask).cuda()
        for k,v in data.items():
          if not(isinstance(v,(list))):
            data[k] = v.cuda()

        img = F.interpolate(
                data['image'].float(),
                size=(1024, 2048),
                mode='nearest'
            ).squeeze()

        img = inv_normalize(img.unsqueeze(0))
        img = img.cpu().numpy()
        img = img.transpose((1,2,0))
        img = (img-np.min(img))/(np.max(img) - np.min(img))

        img = img*255
        img = img.astype(np.uint8)
        img_pil = Image.fromarray(img)

        out = efficientps(data)
        #add the solo outputs to the efficientps output
        out['solo'] = solo_mask
        panoptic_result = panoptic_segmentation_module(cfg,
            out,
            'cuda')

        result_l.append({'panoptic':panoptic_result, 'image_id':data['image_id']})

        generate_pred_panoptic(cfg, result_l)

        img_panoptic = F.interpolate(
                panoptic_result.unsqueeze(0).float(),
                size=(1024, 2048),
                mode='nearest'
            )[0,0,...]
        img_panoptic = img_panoptic.cpu().numpy()

        mask = id2rgb(img_panoptic)

        img_mask = Image.fromarray(mask)
        img_mask.save(os.path.join('/content/drive/MyDrive/EfficientPS/Outputs', str(i)+'panopticpred.png'))
        img_mask.putalpha(200)
        img_pil.paste(img_mask, (0,0), mask = img_mask.convert('L'))
        img_pil.save(os.path.join('/content/drive/MyDrive/EfficientPS/Outputs', str(i)+'output.png'))

  generate_pred_panoptic(cfg, result_l)
  pq_res = pq_compute(
      gt_json_file= os.path.join(cfg.DATASET_PATH,
                                  cfg.VALID_JSON),
      pred_json_file= os.path.join(cfg.DATASET_PATH,
                                    cfg.PRED_JSON),
      gt_folder= os.path.join(cfg.DATASET_PATH,
                              "gtFine/cityscapes_panoptic_val/"),
      pred_folder=os.path.join(cfg.DATASET_PATH, cfg.PRED_DIR)
  )
  print("PQ", 100 * pq_res["All"]["pq"])
  print("SQ", 100 * pq_res["All"]["sq"])
  print("RQ", 100 * pq_res["All"]["rq"])
  print("PQ_th", 100 * pq_res["Things"]["pq"])
  print("SQ_th", 100 * pq_res["Things"]["sq"])
  print("RQ_th", 100 * pq_res["Things"]["rq"])
  print("PQ_st", 100 * pq_res["Stuff"]["pq"])
  print("SQ_st", 100 * pq_res["Stuff"]["sq"])
  print("RQ_st", 100 * pq_res["Stuff"]["rq"])


if __name__ == "__main__":
  main()



