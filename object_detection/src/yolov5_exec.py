#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import time
import json
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

# define path
this_path = os.path.dirname(os.path.abspath(__file__))
top_path = os.path.dirname(this_path)
sys.path.append(this_path)

# Add library path
sys.path.insert(0, "{}/yolov5/".format(top_path))

from utils.datasets import letterbox
from utils.utils import torch_utils, non_max_suppression, scale_coords

from common.log import Log
from common.utils import image_loader
from common.config import read_config
from common.image import plot_bbox
from common.box_utils import box_iou


# 定数
IMAGE_DIR = "./img"
SAVE_DIR = "result/yolov5"
RESULT_FILE = "yolov5_result.csv"


logger = Log.getLogger("yolo", no_handler=True)


def main():
    # load cofig
    config = read_config("config.yaml", os.path.join(top_path, "cfg", "yolov5"))
    ans_conv = read_config("ans_conv.yaml", os.path.join(top_path, "cfg"))
    conv_info = read_config("conv_info.yaml", os.path.join(top_path, "cfg"))
    # conv_info = { k.lower():v.lower() for k,v in config.get("conv", {}).items() }
    weights = config["weights"]
    device = config["device"]
    img_size = 640
    
    thre_score = config["thre_score"]
    thre_iou = config["thre_iou"]
    
    logger.info(json.dumps(config, indent=2))
    
    device = torch_utils.select_device(device)
    model = torch.load(weights, map_location=device)['model']
    model.to(device).eval()
    
    # Half precision
    half = False
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
    
    # names
    names = model.names if hasattr(model, 'names') else model.modules.names
    
    st = time.time()
    result = []
    images = image_loader(IMAGE_DIR)
    num = 0
    for file_name, file_path, df_box in images:
        print("[file]", file_name)
        
        try:
            img0 = cv2.imread(file_path)
            
            # Padded resize
            img = letterbox(img0, new_shape=img_size)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment=False)[0]

            # to float
            if half:
                pred = pred.float()
            
            pred = non_max_suppression(
                    pred, conf_thres=thre_score, iou_thres=0.5,
                    fast=True,
                    )

            labels = []
            scores = []
            boxes = []
            det = pred[0]
            if det is not None and 0 < len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # # Print results
                # s = ""
                # for c in det[:, -1].unique():
                    # n = (det[:, -1] == c).sum()  # detections per class
                    # s += '%g %ss, ' % (n, names[int(c)])
                # print(s)

                for *xyxy, conf, cls in det:
                    labels.append(names[int(cls)])
                    scores.append(conf)
                    boxes.append([ v.item() for v in xyxy ])
        
            # result data
            df_res = pd.DataFrame()
            df_res["name"] = [ x.lower() for x in labels ]
            df_res["label"] = df_res.name.apply(lambda x: conv_info.get(x, x))
            df_res["score"] = scores
            df_res["x"] = [ b[0] for b in boxes ]
            df_res["y"] = [ b[1] for b in boxes ]
            df_res["x2"] = [ b[2] for b in boxes ]
            df_res["y2"] = [ b[3] for b in boxes ]
            df_res["box"] = boxes
            df_res["eval"] = 0
                
            print(df_res[["name","score","x","y","x2","y2"]])
            
            # bbox plot
            bbox = []
            for row in df_res.itertuples():
                x, y, x2, y2 = row.box
                bbox.append([x,y,x2,y2])
            
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            plot_bbox(img0, bbox, df_res.score, df_res.name)
            
            # save result image
            plt.savefig(
                os.path.join(SAVE_DIR, file_name),
                bbox_inches='tight', pad_inches=0
                )
            plt.close()
            
            # check result
            df_box["label"] = df_box.label.apply(lambda x: ans_conv.get(x, x))
            for label in df_box.label.unique():
                df_a = df_box[df_box.label==label]
                df_b = df_res[df_res.label==label]
                box_a = df_a[["x","y","x2","y2"]].values
                box_b = df_b[["x","y","x2","y2"]].values
                
                # IoU
                iou = box_iou(box_a, box_b) if 0 < len(df_b) else None
                
                for i, row in enumerate(df_a.itertuples()):
                    n = iou[i].argmax() if 0 < len(df_b) else -1
                    match = -1 < n and 0 < iou[i,n] and iou[:,n].argmax() == i
                    ok = match and thre_iou < iou[i,n]
                    box = df_b.iloc[n].box if match else None
                    
                    result.append([
                        file_path,  # ファイルパス
                        label,      # 正解ラベル
                        1 if ok else 0, # 正誤
                        df_b.iloc[n].score if match else None,
                        iou[i,n] if match else None,
                        row.x,
                        row.y,
                        row.x2,
                        row.y2,
                        box[0] if match else None,
                        box[1] if match else None,
                        box[2] if match else None,
                        box[3] if match else None,
                        df_b.iloc[n]["name"] if match else None,
                    ])
                    
                    if match:
                        df_res.loc[df_b.iloc[[n]].index,"eval"] = 1
                
            # 未使用の結果を保存
            for row in df_res.itertuples():
                if row.eval == 1:
                    continue
                    
                result.append([
                    file_path,  # ファイルパス
                    "-",   # 正解ラベル
                    None,   # 正誤
                    row.score,
                    None,
                    None,
                    None,
                    None,
                    None,
                    row.box[0],
                    row.box[1],
                    row.box[2],
                    row.box[3],
                    row.name,
                ])
            
            num += 1
            # break
        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exc()
            logger.warn("failed to analysis : {}".format(file_path))
            break
    
    lap = time.time() - st
    
    result = pd.DataFrame(
        result,
        columns=[
            "file_path", "label", "ok",
            "score", "IoU",
            "x", "y", "x2", "y2",
            "_x", "_y", "_x2", "_y2", "_label"
        ])
    
    logger.info("画像処理時間 : {}".format(lap))
    logger.info("画像枚数 : {}".format(num))
    if 0 < num:
        logger.info("画像一枚あたり : {}".format(lap/num))

    result.to_csv(RESULT_FILE, index=False)


if __name__ == "__main__":
    with torch.no_grad():
        main()
