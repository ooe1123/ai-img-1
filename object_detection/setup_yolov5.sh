#!/bin/sh


### Build 
rm -rf yolov5
git clone https://github.com/ultralytics/yolov5

### weightsダウンロード
(cd yolov5 && source weights/download_weights.sh)
# curl -L "https://drive.google.com/uc?export=download&id=1R5T6rIyy3lLwgFXNms8whc-387H0tMQO" -o yolov5/weights/yolov5s.pt
# curl -L "https://drive.google.com/uc?export=download&id=1vobuEExpWQVpXExsJ2w-Mbf3HJjWkQJr" -o yolov5/weights/yolov5m.pt
# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1hrlqD1Wdei7UT4OgT785BEk1JwnSvNEV" > /dev/null
# curl -b ./cookie -L "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1hrlqD1Wdei7UT4OgT785BEk1JwnSvNEV" -o yolov5/weights/yolov5l.pt
# curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1mM8aZJlWTxOg7BZJvNUMrTnA2AbeCVzS" > /dev/null
# curl -b ./cookie -L "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1mM8aZJlWTxOg7BZJvNUMrTnA2AbeCVzS" -o yolov5/weights/yolov5x.pt
# rm -rf ./cookie