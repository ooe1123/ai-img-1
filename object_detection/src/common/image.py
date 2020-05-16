#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os

import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageColor


def plot_bbox(img, bbox, scores, label_names):
    """
    物体検出の予測結果を画像で表示させる関数。

    Parameters
    ----------
    img:ndarray画像
        対象の画像データ
    bbox: list
        物体のBBoxのリスト
    scores: list
        物体の確信度。
    label_names: list
        ラベル名の配列

    Returns
    -------
    なし。rgb_imgに物体検出結果が加わった画像が表示される。
    """

    label_index = { x:i for i,x in enumerate(set(label_names)) }

    # 枠の色の設定
    num_classes = len(label_index)  # クラス数（背景のぞく）
    # colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    colors = [
        [ float(int(x,16))/255 for x in [s[1:3],s[3:5],s[5:7]] ]
            for s in ImageColor.colormap.values() if isinstance(s, str)
        ]

    # 画像の表示
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    currentAxis = plt.gca()

    # BBox分のループ
    for i, bb in enumerate(bbox):

        # ラベル名
        label_name = label_names[i]
        # color = colors[label_index[label_name]]  # クラスごとに別の色の枠を与える
        color = colors[hash(label_name) % len(colors)]  # クラスごとに別の色の枠を与える

        # 枠につけるラベル　例：person;0.72　
        if scores is not None:
            sc = scores[i]
            display_txt = '%s: %.2f' % (label_name, sc)
        else:
            display_txt = '%s' % (label_name)

        # 枠の座標
        xy = (bb[0], bb[1])
        width = bb[2] - bb[0]
        height = bb[3] - bb[1]

        # 長方形を描画する
        currentAxis.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor=color, linewidth=2))

        # 長方形の枠の左上にラベルを描画する
        currentAxis.text(xy[0], xy[1], display_txt, bbox={
                         'facecolor': color, 'alpha': 0.5})
