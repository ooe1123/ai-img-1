#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import time
import csv
import glob
import json
import traceback

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openvino.inference_engine import IECore

# define path
this_path = os.path.dirname(os.path.abspath(__file__))
top_path = os.path.dirname(this_path)
sys.path.append(this_path)

from common.log import Log
from common.utils import image_loader
from common.config import read_config
from common.image import plot_bbox
from common.box_utils import box_iou


# 定数
IMAGE_DIR = "./img"
SAVE_DIR = "result/openvino"
RESULT_FILE = "openvino_result.csv"


logger = Log.getLogger("OpenVino")


def openvino_exec():
    # load cofig
    config = read_config("config.yaml", os.path.join(top_path, "cfg", "openvino"))
    ans_conv = read_config("ans_conv.yaml", os.path.join(top_path, "cfg"))
    conv_info = read_config("conv_info.yaml", os.path.join(top_path, "cfg"))
    
    model_xml = os.path.join("cfg", config["model"])
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    
    label_list = read_config(config["label_file"], os.path.join(top_path, "cfg", "openvino"))
    thre_score = config["thre_score"]
    thre_iou = config["thre_iou"]
    
    device = config["device"].upper()
    ie = IECore()
    net = ie.read_network(model=model_xml, weights=model_bin)
    
    # ------------
    versions = ie.get_versions(device)
    print("Device info:")
    print("{}{}".format(" " * 8, device))
    print("{}MKLDNNPlugin version ......... {}.{}".format(" " * 8, versions[device].major,
                                                          versions[device].minor))
    print("{}Build ........... {}".format(" " * 8, versions[device].build_number))
    
    if "CPU" in device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            logger.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(device, ', '.join(not_supported_layers)))
            logger.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)
    
    # ---- Read and preprocess input ----
    print("inputs number: " + str(len(net.inputs.keys())))
    
    for input_key in net.inputs:
        print("input shape: " + str(net.inputs[input_key].shape))
        print("input key: " + input_key)
        if len(net.inputs[input_key].layout) == 4:
            n, c, h, w = net.inputs[input_key].shape
    
    # ---- Prepare input blobs ----
    logger.info("Preparing input blobs")
    assert (len(net.inputs.keys()) == 1 or len(
        net.inputs.keys()) == 2), "Sample supports topologies only with 1 or 2 inputs"
    out_blob = next(iter(net.outputs))
    input_name, input_info_name = "", ""
    
    for input_key in net.inputs:
        if len(net.inputs[input_key].layout) == 4:
            input_name = input_key
            logger.info("Batch size is {}".format(net.batch_size))
            net.inputs[input_key].precision = 'U8'
        elif len(net.inputs[input_key].layout) == 2:
            input_info_name = input_key
            net.inputs[input_key].precision = 'FP32'
            if net.inputs[input_key].shape[1] != 3 and net.inputs[input_key].shape[1] != 6 or \
                net.inputs[input_key].shape[0] != 1:
                logger.error('Invalid input info. Should be 3 or 6 values length.')
    
    data = {}
    if input_info_name != "":
        infos = np.ndarray(shape=(n, c), dtype=float)
        for i in range(n):
            infos[i, 0] = h
            infos[i, 1] = w
            infos[i, 2] = 1.0
        data[input_info_name] = infos
    
    # ---- Prepare output blobs ----
    logger.info('Preparing output blobs')
    
    output_name, output_info = "", net.outputs[next(iter(net.outputs.keys()))]
    for output_key in net.outputs:
        if net.layers[output_key].type == "DetectionOutput":
            output_name, output_info = output_key, net.outputs[output_key]
    
    if output_name == "":
        logger.error("Can't find a DetectionOutput layer in the topology")
    
    output_dims = output_info.shape
    if len(output_dims) != 4:
        logger.error("Incorrect output dimensions for SSD model")
    max_proposal_count, object_size = output_dims[2], output_dims[3]
    
    if object_size != 7:
        logger.error("Output item should have 7 as a last dimension")
    
    output_info.precision = "FP32"
    
    # ------------
    
    st = time.time()
    result = []
    num = 0
    for file_name, file_path, df_box in image_loader(IMAGE_DIR):
        print("[file]", file_name)
        
        try:
            # Read and pre-process input images
            images = np.ndarray(shape=(n, c, h, w))
            data[input_name] = images
            
            img = _img = cv2.imread(file_path)
            ih, iw = img.shape[:-1]
            if (ih, iw) != (h, w):
                _img = cv2.resize(img, (w, h))
                logger.warning("Image {} is resized from {} to {}".format(file_path, (ih, iw), (h, w)))
            _img = _img.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            images[0] = _img
            
            exec_net = ie.load_network(network=net, device_name=device)
            res = exec_net.infer(inputs=data)
            res = res[out_blob]
            
            labels, scores, boxes = [], [], []
            for i, proposal in enumerate(res[0][0]):
                if proposal[2] > 0:
                    imid = np.int(proposal[0])
                    label = label_list[np.int(proposal[1])]
                    confidence = proposal[2]
                    xmin = np.int(iw * proposal[3])
                    ymin = np.int(ih * proposal[4])
                    xmax = np.int(iw * proposal[5])
                    ymax = np.int(ih * proposal[6])
                    box = [
                        xmin, ymin, xmax, ymax,
                    ]
                    # print("[{},{}] element, prob = {:.6}    ({},{})-({},{}) batch id : {}" \
                          # .format(i, label, confidence, xmin, ymin, xmax, ymax, imid))
                    
                    labels.append(label)
                    scores.append(confidence)
                    boxes.append(box)
            
            df_res = pd.DataFrame()
            df_res["name"] = [x.lower() for x in labels]
            df_res["label"] = df_res.name.apply(lambda x: conv_info.get(x, x))
            df_res["score"] = scores
            df_res["x"] = [ b[0] for b in boxes ]
            df_res["y"] = [ b[1] for b in boxes ]
            df_res["x2"] = [ b[2] for b in boxes ]
            df_res["y2"] = [ b[2] for b in boxes ]
            df_res["box"] = boxes
            df_res = df_res[thre_score <= df_res.score].reset_index()
            df_res["eval"] = 0
            
            print(df_res[["name", "score", "x", "y", "x2", "y2"]])
            
            # bbox plot
            bbox = []
            for row in df_res.itertuples():
                x, y, x2, y2 = row.box
                bbox.append([x, y, x2, y2])

            plot_bbox(img, bbox, df_res.score, df_res.name)

            # save result image
            plt.savefig(
                os.path.join(SAVE_DIR, file_name), bbox_inches="tight", pad_inches=0,
            )
            plt.close()

            # check result
            df_box["label"] = df_box.label.apply(lambda x: ans_conv.get(x, x))
            for label in df_box.label.unique():
                df_a = df_box[df_box.label == label]
                df_b = df_res[df_res.label == label]
                box_a = df_a[["x", "y", "x2", "y2"]].values
                box_b = df_b[["x", "y", "x2", "y2"]].values

                # IoU
                iou = box_iou(box_a, box_b) if 0 < len(df_b) else None

                for i, row in enumerate(df_a.itertuples()):
                    n = iou[i].argmax() if 0 < len(df_b) else -1
                    match = -1 < n and 0 < iou[i, n] and iou[:, n].argmax() == i
                    ok = match and thre_iou < iou[i, n]
                    box = df_b.iloc[n].box if match else None

                    result.append(
                        [
                            file_path,  # ファイルパス
                            label,  # 正解ラベル
                            1 if ok else 0,  # 正誤
                            df_b.iloc[n].score if match else None,
                            iou[i, n] if match else None,
                            row.x,
                            row.y,
                            row.x2,
                            row.y2,
                            box[0] if match else None,
                            box[1] if match else None,
                            box[2] if match else None,
                            box[3] if match else None,
                            df_b.iloc[n]["name"] if match else None,
                        ]
                    )

                    if match:
                        df_res.loc[df_b.iloc[[n]].index, "eval"] = 1

            # 未使用の結果を保存
            for row in df_res.itertuples():
                if row.eval == 1:
                    continue

                result.append(
                    [
                        file_path,  # ファイルパス
                        "-",  # 正解ラベル
                        None,  # 正誤
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
                    ]
                )

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

    lap = time.time() - st
    logger.info("画像処理時間 : {}".format(lap))
    logger.info("画像枚数 : {}".format(num))
    if 0 < num:
        logger.info("画像一枚あたり : {}".format(lap/num))

    result.to_csv(RESULT_FILE, index=False)


if __name__ == "__main__":
    st = time.time()
    logger.info("[ Start ]")
    openvino_exec()
    logger.info("実行時間 : {}".format(time.time() - st))
    logger.info("[  End  ]")
