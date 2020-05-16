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

# define path
this_path = os.path.dirname(os.path.abspath(__file__))
top_path = os.path.dirname(this_path)
sys.path.append(this_path)

from common.log import Log
from common.utils import image_loader
from common.image import plot_bbox


# 定数
IMAGE_DIR = "./img"
SAVE_DIR = "result/bbox"


def main():
    images = image_loader(IMAGE_DIR)
    for file_name, file_path, df_box in images:
        print("[file]", file_name)
        
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        h, w, _ = img.shape
        
        # bbox plot
        bbox = []
        for row in df_box.itertuples():
            bbox.append([row.x, row.y, row.x+row.w, row.y+row.h])
        
        plot_bbox(img, bbox, None, df_box.label)
        
        # save result image
        plt.savefig(
            os.path.join(SAVE_DIR, file_name),
            bbox_inches='tight', pad_inches=0
            )
        # plt.show()
        plt.close()


if __name__ == "__main__":
    main()
