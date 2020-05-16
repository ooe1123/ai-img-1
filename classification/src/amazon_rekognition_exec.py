#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import time
import configparser
import csv
import glob
import json
import traceback

import boto3

# define path
this_path = os.path.dirname(os.path.abspath(__file__))
top_path = os.path.dirname(this_path)
sys.path.append(this_path)

from common.utils import image_loader
from common.config import read_config
from common.log import Log


# 定数
WORK_DIR = "./amazon_rekognition_result"
RESULT_FILE = "amazon_rekognition_result.csv"
BUCKET_NAME = "ai-sample-img"


logger = Log.getLogger("AmazonExec")


def amazon_rekognition_exec():
    # load cofig
    config = read_config("config.yaml", os.path.join(top_path, "cfg", "amazon_cfg"))
    conv_info = {k.lower(): v.lower() for k, v in config.get("conv", {}).items()}

    logger.info(json.dumps(config, indent=2))

    """AmazonRekognitionで画像分類して、結果をJSONへ出力"""

    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)

    # Amazon Rekognition
    # session = boto3.Session()
    session = boto3.Session(
        aws_access_key_id=config["aws_access_key_id"],
        aws_secret_access_key=config["aws_secret_access_key"],
        region_name=config["region_name"],
    )
    rekognition = session.client("rekognition")
    # S3
    s3 = session.resource("s3")
    s3.create_bucket(
        Bucket=BUCKET_NAME,
        CreateBucketConfiguration={"LocationConstraint": config["region_name"]},
    )
    bucket = s3.Bucket(BUCKET_NAME)

    # 画像のパスを取得
    image_paths = image_loader(os.path.join(top_path, "img"))

    st = time.time()
    json_files = []
    for label, image_path in image_paths:
        try:
            logger.info("target local image : [{}]".format(image_path))

            # タグ、ファイル名拡張子、ファイル名
            tag, img_name, name = get_image_info(image_path)
            s3_path = tag + "/" + img_name
            # S3 upload
            bucket.upload_file(str(image_path), s3_path)
            logger.info("S3 Upload [{}] -> [{}]".format(image_path, s3_path))

            # 分析
            res = rekognition.detect_labels(
                Image={"S3Object": {"Bucket": BUCKET_NAME, "Name": s3_path}}, MaxLabels=12
            )
            res["ImageFile"] = image_path
            res["Answer"] = label
            json_file = os.path.join(WORK_DIR, "{}_{}.json".format(tag, name))
            with open(json_file, "w") as f:
                json.dump(res, f, indent=4)
                logger.info("Write json file. [{}]".format(json_file))
            json_files.append(json_file)
        except Exception:
            traceback.print_exc()
            logger.warn("failed to analysis : {}".format(image_path))

    lap = time.time() - st
    logger.info("画像処理時間 : {}".format(lap))
    logger.info("画像枚数 : {}".format(len(json_files)))
    logger.info("画像一枚あたり : {}".format(lap / len(json_files)))

    # S3 bucketと画像削除
    logger.info("S3 image delete.")
    for key in bucket.objects.all():
        key.delete()
        logger.info(" - S3 Delete key: {}".format(key))
    bucket.delete()

    # Result.csv出力
    logger.info("Write result.csv file.")
    create_result_file(json_files, conv_info)


def get_image_info(image_path):
    """画像のパスを整形する"""
    # 画像ファイルが入ってるディレクトリ名をタグとして扱う
    tag = os.path.basename(os.path.dirname(image_path))
    # ファイル名（拡張子あり）
    img_name = os.path.basename(image_path)
    # ファイル名（拡張子なし）
    name = os.path.splitext(img_name)[0]
    return tag, img_name, name


def create_result_file(json_files, conv_info):
    """JSONファイルを集計して結果ファイルに出力する"""
    header = [
        "file_path",
        "label",
        "ok",
        "pred1",
        "score1",
        "pred2",
        "score2",
        "pred3",
        "score3",
        "pred4",
        "score4",
        "pred5",
        "score5",
        "pred6",
        "score6",
        "pred7",
        "score7",
        "pred8",
        "score8",
        "pred9",
        "score9",
        "pred10",
        "score10",
        "pred11",
        "score11",
        "pred12",
        "score12",
    ]
    N = len(header) - 3
    with open(RESULT_FILE, "w", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(header)
        for j in json_files:
            tag = os.path.basename(j).split("_")[0].lower()
            labels = []
            is_correct = False
            with open(j, "r") as f:
                df = json.load(f)
                image_path = df["ImageFile"]
                answer = df["Answer"]
                for dd in sorted(
                        df["Labels"],
                        key=lambda x: x["Confidence"],
                        reverse=True):
                    name = dd["Name"].lower()

                    labels.append(name)
                    labels.append(dd["Confidence"])
                    if conv_info.get(name, name) == answer:
                        is_correct = True

            labels.extend([""] * N)  # padding

            row = [image_path, answer, 1 if is_correct else 0]
            row.extend(labels[:N])
            w.writerow(row)


# def test_s3():
#     # S3
#     s3 = boto3.resource("s3")
#     # bucket作成
#     s3.create_bucket(Bucket="ai-sample-img")
#     bucket = s3.Bucket("ai-sample-img")
#     # アップロード
#     bucket.upload_file("../../img/dog/sample_husky2.jpg", "dog/sample_husky2.jpg")
#     # 画像削除
#     for key in bucket.objects.all():
#         key.delete()
#         logger.info("S3 Delete key: {}".format(key))
#     # bucket削除
#     bucket.delete()


if __name__ == "__main__":
    st = time.time()
    logger.info("[ Start ]")
    amazon_rekognition_exec()
    logger.info("実行時間 : {}".format(time.time() - st))
    logger.info("[  End  ]")

    # test_s3()
