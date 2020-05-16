#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import glob
from collections import namedtuple
import re


def image_loader(path):
    for file_path in glob.glob("{}/*/*".format(path)):
        a = re.split(r"[/\\]", file_path)
        label = a[-2].lower()
        yield (label, file_path)
    
