classification
====

## yolo

#### セットアップ
darknetのリポジトリをcloneしてきてビルド、学習済み重みファイルのダウンロードを行うスクリプトを実行する。
```
# source setup_yolo.sh
```
もしGPUのない環境であるなら、setup_yolo.shの以下の行をコメントアウトしておく。
```
sed "s/GPU=0/GPU=1/" -i darknet/Makefile
```

#### 実行
```
# python src/yolo_exec.py
```

## Amazon Rekognition

#### 事前準備
Amazon RekognitionとS3のアクセス権限を設定したアクセスキーを設定ファイルに記述する。

1. AWSコンソールのIAMリソースから、次の2つのポリシーを適用したユーザを作成する。
   「ユーザ」 > 「アクセス権限」
* AmazonS3FullAccess
* AmazonRekognitionFullAccess

2. 「ユーザ」 > 「認証情報」から、アクセスキーを作成する。

3. cfg/amazon_rekognition/config.yaml に、アクセスキーとシークレットアクセスキーを設定する。
```
aws_access_key_id: XXX
aws_secret_access_key: XXX
```

#### 実行
```
# python src/amazon_rekognition_exec.py
```

## Watson Visual Recognition

#### 事前準備
Visual RecognitionのAPIキーとURLを設定ファイルに記述する。

1. IBM Cloudの管理画面から、Visual Recognitionサービスを起動する。

2. IBM Cloudのリソース管理画面から起動したVisual Recognitionサービスを選択し、APIキーとURLの情報を確認する。

3. cfg/watson_cfg/config.yaml に、APIキーとURLを設定する。
```
apikey: XXX
url: https://api.xxx.visual-recognition.watson.cloud.ibm.com
```

#### 実行
```
# python src/watson_exec.py
```

## OpenVINO

#### セットアップ
OpenVINOのパッケージをダウンロードしてきてインストール＆学習済み重みファイルのダウンロードを行うスクリプトを実行する。
```
# source setup_openvino.sh
```
途中質問に対しては適宜回答を入力する
```
This Agreement forms a legally binding contract between you, or the company or
other legal entity ('Legal Entity') for which you represent and warrant that you
have the legal authority to bind that Legal Entity, (each, 'You' or 'Your') and
Intel Corporation and its subsidiaries (collectively 'Intel') regarding Your use
of the Materials. By downloading, installing, copying or otherwise using the
Materials, You agree to be bound by the terms of this Agreement. If You do not
agree to the terms of this Agreement, do not download, install, copy or
otherwise use the Materials. You affirm that You are 18 years old or older or,
--------------------------------------------------------------------------------
Type "accept" to continue or "decline" to go back to the previous menu: accept
```
```
--------------------------------------------------------------------------------

   1. I consent to the collection of my Information
   2. I do NOT consent to the collection of my Information

   b. Back
   q. Quit installation

--------------------------------------------------------------------------------
Please type a selection: 1
```
インストール先のパスの違いでエラーがでたら、OPENCV_HOME変数を適切なパスに変えて以下をやり直す
```
export OPENCV_HOME=/home/ubuntu/intel/openvino_2019.3.376
(cd $OPENCV_HOME/install_dependencies/ && sudo ./install_openvino_dependencies.sh)
source $OPENCV_HOME/bin/setupvars.sh
```

#### 実行
```
# python src/openvino_exec.py
```
