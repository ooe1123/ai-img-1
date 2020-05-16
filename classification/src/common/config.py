#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import yaml


def read_config(file_name, path_dir="."):
    with open(os.path.join(path_dir, file_name)) as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config
    
