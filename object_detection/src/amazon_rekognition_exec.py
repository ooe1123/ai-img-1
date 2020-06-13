#!/usr/bin/env python
import configparser
import csv
import glob
import json
import os
import sys
import time
import traceback

import boto3
import cv2
import numpy
import matplotlib.pyplot as plt
import pandas as pd

from common.box_utils import box_iou
from common.config import read_config
from common.image import plot_bbox
from common.log import Log
from common.utils import image_loader

# define path
this_path = os.path.dirname(os.path.abspath(__file__))
top_path = os.path.dirname(this_path)
sys.path.append(this_path)


# 定数
IMAGE_DIR = "./img"
SAVE_DIR = "result/amazon"
WORK_DIR = "./amazon_rekognition_result"
RESULT_FILE = "amazon_rekognition_result.csv"


logger = Log.getLogger("AmazonExec")


def amazon_rekognition_exec():
    """AmazonRekognitionで画像分類して、結果をJSONへ出力"""

    # load cofig
    config = read_config("config.yaml", os.path.join(top_path, "cfg", "amazon"))
    bucket_name = config["bucket_name"]
    exec_rekognition = config["exec_rekognition"]
    logger.info(json.dumps(config, indent=2))

    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)

    # 画像のパスを取得
    images = image_loader(IMAGE_DIR)
    if exec_rekognition:
        # Amazon Rekognition
        session = boto3.Session(
            aws_access_key_id=config["aws_access_key_id"],
            aws_secret_access_key=config["aws_secret_access_key"],
            region_name=config["region_name"],
        )
        rekognition = session.client("rekognition")

        # S3 create
        logger.info("create S3 bucket: %s." % bucket_name)
        create_flg = False
        s3 = session.resource("s3")
        try:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": config["region_name"]},
            )
            create_flg = True
        except Exception as e:
            if e.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
                raise
            
        bucket = s3.Bucket(bucket_name)

        # Amazon Rekognition分析
        st = time.time()
        anno_list, num = aws_reko(bucket, rekognition, images)
        lap = time.time() - st
        logger.info("画像処理時間 : {}".format(lap))
        logger.info("画像枚数 : {}".format(num))
        if 0 < len(anno_list):
            logger.info("画像一枚あたり : {}".format(lap / num))

        # S3 bucketと画像削除
        logger.info("S3 image delete.")
        # for key in bucket.objects.all():
            # key.delete()
            # logger.info(" - S3 Delete key: {}".format(key))
        # bucket.objects.all().delete()
        if create_flg:
            logger.info("delete S3 bucket: %s." % bucket_name)
            bucket.delete()
    else:
        def gene_anno(images):
            for file_name, file_path, df_box in images:
                json_file = "{}.json".format(file_name.split(".")[0])
                yield (file_name, file_path, json_file, df_box)
        
        anno_list = gene_anno(images)

    create_result_file(anno_list, config)


def aws_reko(bucket, rekognition, img_list):
    anno_list = []
    num = 0
    for file_name, file_path, df_box in img_list:
        logger.info("target local image : [{}]".format(file_path))
        json_file = "{}.json".format(file_name.split(".")[0])

        try:
            s3_path = "img/{}".format(file_name)
            # S3 upload
            # bucket.upload_file(str(file_path), s3_path)
            logger.info("S3 Upload [{}] -> [{}]".format(file_path, s3_path))

            # 分析
            res = rekognition.detect_labels(
                Image={"S3Object": {"Bucket": bucket.name, "Name": s3_path}},
                MaxLabels=12,
            )

            # JSON 書き出し
            result_file = os.path.join(WORK_DIR, json_file)
            with open(result_file, "w") as f:
                json.dump(res, f, indent=4)
                logger.info("Write json file. [{}]".format(result_file))
            
            anno_list.append((file_name, file_path, json_file, df_box))
            num += 1
        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exc()
            logger.warn("failed to analysis : {}".format(file_path))
            break

    return anno_list, num


def create_result_file(anno_list, config):
    ans_conv = read_config("ans_conv.yaml", os.path.join(top_path, "cfg"))
    conv_info = read_config("conv_info.yaml", os.path.join(top_path, "cfg"))
    thre_score = config["thre_score"]
    thre_iou = config["thre_iou"]
    
    result = []
    for file_name, file_path, json_file, df_box in anno_list:
        print("[file]", file_name)

        try:
            result_file = os.path.join(WORK_DIR, json_file)

            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            h, w, _ = img.shape
            
            with open(result_file, "r") as f:
                res_json = json.load(f)
            
            df_res = list_objects(res_json, h, w)
            df_res["label"] = df_res.name.apply(lambda x: conv_info.get(x, x))
            df_res = df_res[thre_score <= df_res.score].reset_index()
            
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

        except Exception:
            traceback.print_exc()
            continue

    result = pd.DataFrame(
        result,
        columns=[
            "file_path", "label", "ok",
            "score", "IoU",
            "x", "y", "x2", "y2",
            "_x", "_y", "_x2", "_y2", "_label"
        ])
    
    result.to_csv(RESULT_FILE, index=False)


def list_objects(res_json, h, w):
    labels, scores, boxes = [], [], []
    
    for l in res_json["Labels"]:
        name = l["Name"]
        score = l["Confidence"]
        for x in l["Instances"]:
            b = x["BoundingBox"]
            min_score = min(score, x["Confidence"])
            box = [
                b["Left"] * w,
                b["Top"] * h,
                (b["Left"] + b["Width"]) * w,
                (b["Top"] + b["Height"]) * h,
            ]
            labels.append(name)
            scores.append(min_score)
            boxes.append(box)

    df_res = pd.DataFrame()
    df_res["name"] = [x.lower() for x in labels]
    df_res["score"] = scores
    df_res["x"] = [ b[0] for b in boxes ]
    df_res["y"] = [ b[1] for b in boxes ]
    df_res["x2"] = [ b[2] for b in boxes ]
    df_res["y2"] = [ b[3] for b in boxes ]
    df_res["box"] = boxes
    df_res["eval"] = 0
    return df_res


if __name__ == "__main__":
    st = time.time()
    logger.info("[ Start ]")
    amazon_rekognition_exec()
    logger.info("実行時間 : {}".format(time.time() - st))
    logger.info("[  End  ]")
