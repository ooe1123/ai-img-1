import time
import json
import os
import sys
import csv
import traceback

from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import VisualRecognitionV3

from common.config import read_config
from common.log import Log
from common.utils import image_loader

# define path
this_path = os.path.dirname(os.path.abspath(__file__))
top_path = os.path.dirname(this_path)
sys.path.append(this_path)


# 定数
WORK_DIR = "./watson_image_recognition_result"
RESULT_FILE = "watson_image_recognition_result.csv"

logger = Log.getLogger("WatsonExec")


def watson_visual_recognition_exec():
    """WatsonVisualRecognitionで画像分類して、結果をJSONへ出力"""

    # load cofig
    config = read_config("config.yaml", os.path.join(top_path, "cfg", "watson_cfg"))
    conv_info = {k.lower(): v.lower() for k, v in config.get("conv", {}).items()}

    logger.info(json.dumps(config, indent=2))

    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)

    authenticator = IAMAuthenticator(config["apikey"])
    visual_recognition = VisualRecognitionV3(
        version=config["version"], authenticator=authenticator
    )
    visual_recognition.set_service_url(config["url"])

    # 画像のパスを取得
    image_paths = image_loader(os.path.join(top_path, "img"))

    st = time.time()
    json_files = []
    for label, image_path in image_paths:
        try:
            logger.info("target local image : [{}]".format(image_path))

            # タグ、ファイル名拡張子、ファイル名
            tag, img_name, name = get_image_info(image_path)

            with open(image_path, "rb") as images_file:
                res = visual_recognition.classify(
                    images_file=images_file,
                    threshold="0.6"
                    # , owners=["me"]
                ).get_result()
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
            labels = []
            is_correct = False
            with open(j, "r") as f:
                df = json.load(f)
                image_path = df["ImageFile"]
                answer = df["Answer"]
                for dd in sorted(
                        df["images"][0]["classifiers"][0]["classes"],
                        key=lambda x: x["score"],
                        reverse=True):
                    name = dd["class"].lower()
                    labels.append(name)
                    labels.append(dd["score"])
                    if conv_info.get(name, name) == answer:
                        is_correct = True

            labels.extend([""] * N)  # padding

            row = [image_path, answer, 1 if is_correct else 0]
            row.extend(labels[:N])
            w.writerow(row)


if __name__ == "__main__":
    st = time.time()
    logger.info("[ Start ]")
    watson_visual_recognition_exec()
    logger.info("実行時間 : {}".format(time.time() - st))
    logger.info("[  End  ]")
