#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import time
import argparse
import glob
import shutil
import json

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--img-dir",
        default="img",
        help="specify the Image Directory"
        )
    parser.add_argument(
        "anno_file",
        help="specify the annotation_file"
        )
    args = parser.parse_args()
    
    with open(args.anno_file) as f:
        info = json.load(f)
    
    img_dir = args.img_dir
    img_info = info["images"]
    anno_info = info["annotations"]
    cat_info = info["categories"]
    
    img_info = { d["id"]:d for d in img_info }
    cat_info = { d["id"]:d["name"] for d in cat_info }
    
    df = pd.DataFrame()
    df["image_id"] = [ d["image_id"] for d in anno_info ]
    df["category_id"] = [ d["category_id"] for d in anno_info ]
    df["bbox"] = [ d["bbox"] for d in anno_info ]
    df = df.sort_values(by=["image_id", "category_id"])
    
    def func(df):
        image_id = df.iloc[0].image_id
        
        if not image_id in img_info:
            return
        
        info = img_info[image_id]
        file_name = info["file_name"]
        h = info["height"]
        w = info["width"]
        
        if not os.path.exists(os.path.join(img_dir, file_name)):
            return
        
        print(file_name)
        
        data = [[],[],[],[],[]] # label, x, y, w, h
        for row in df.itertuples():
            label = cat_info[row.category_id]
            bbox = row.bbox
            data[0].append(label)
            data[1].append(bbox[0])
            data[2].append(bbox[1])
            data[3].append(bbox[2])
            data[4].append(bbox[3])
        
        df = pd.DataFrame(np.array(data).T)
        df.to_csv(
            os.path.join(img_dir, "%s.tsv" % file_name.split(".")[0]),
            sep="\t",
            index=False,
            header=False,
            )
    
    df.groupby("image_id", as_index=False).apply(func)


if __name__ == "__main__":
    main()
