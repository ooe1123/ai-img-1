#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import glob
from collections import namedtuple
import re

import pandas as pd


def image_loader(path):
    for file_path in sorted(glob.glob("{}/*".format(path))):
        if file_path.endswith(".tsv"):
            continue
        
        a = re.split(r"[/\\]", file_path)
        file_name = a[-1].lower()
        
        box_list = []
        
        txt_file = ".".join(file_path.split(".")[:-1]+["tsv"])
        if os.path.exists(txt_file):
            with open(txt_file) as f:
                for line in f.readlines():
                    label, x, y, w, h = line.split("\t")
                    box_list.append([label.lower()] + list(float(v) for v in [x, y, w, h]))

        df_box = pd.DataFrame(box_list, columns=["label","x","y","w","h"])
        df_box["x2"] = df_box[["x","w"]].apply(lambda x: sum(x), axis=1)
        df_box["y2"] = df_box[["y","h"]].apply(lambda x: sum(x), axis=1)

        yield (file_name, file_path, df_box)
