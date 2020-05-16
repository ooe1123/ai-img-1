#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import shutil

import numpy as np
import pandas as pd


def main():
    info = [
        ("validation", "validation-annotations-human-imagelabels.csv"),
        ("test", "test-annotations-human-imagelabels.csv"),
    ]
    
    df_desc = pd.read_csv("oidv6-class-descriptions.csv")
    id_label = dict(zip(df_desc["LabelName"], df_desc["DisplayName"]))

    for dname, anno_file in info:
        df = pd.read_csv(anno_file, sep=",")
        df = df[df["Confidence"]==1]

        group = df[["ImageID","LabelName"]].groupby("ImageID")
        df_uniq = group.count()
        df_uniq = df_uniq[df_uniq["LabelName"]==1]
        
        df = df[df["ImageID"].isin(df_uniq.index)]
        
        for _, row in df.iterrows():
            file_name = "%s.jpg" % row.ImageID
            path = os.path.join(dname, file_name)
            if not os.path.exists(path):
                continue
            
            label = id_label.get(row.LabelName, None)
            if not label:
                continue
            
            print(path, label)
            
            dst_path = os.path.join("open_image", label, file_name)
            if not os.path.exists(os.path.dirname(dst_path)):
                os.makedirs(os.path.dirname(dst_path))

            shutil.copy(path, dst_path)


if __name__ == "__main__":
    main()
