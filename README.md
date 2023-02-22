# YOLOv4 evaluation

The project evaluates darknet yolov4 using mAP metric from [Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics.git).

## Installation

Clone the project and submodules:
```bash
git clone https://github.com/klymya/yolov4-evaluation.git && \
	cd yolov4-evaluation && git submodule update --init --recursive
```

Install python requirements:
```bash
pip3 install -r Object-Detection-Metrics/requirements.txt
```

## Usage

To evaluate model run `src/eval_yolov4.py`.
```
python3 src/eval_yolov4.py -h
usage: Evaluate yolov4 using mAP from 'Object-Detection-Metrics' project. [-h] --darknet DARKNET --data-file DATA_FILE --cfg CFG --weights WEIGHTS [--width WIDTH] [--height HEIGHT] [--thresh THRESH] [--pascalvoc PASCALVOC] --out OUT

optional arguments:
  -h, --help            show this help message and exit
  --darknet DARKNET     Path to darknet executable file.
  --data-file DATA_FILE
                        Path to yolo's *.data file.
  --cfg CFG             Path to the model *.cfg file.
  --weights WEIGHTS     Path to the model *.weights file.
  --width WIDTH         The model's input width. Default = 512.
  --height HEIGHT       The model's input height. Default = 512.
  --thresh THRESH       Confidence threshold. Default = 0.25.
  --pascalvoc PASCALVOC
                        Path to pascalvoc.py script. Default = Object-Detection-Metrics/pascalvoc.py.
  --out OUT             Path to folder where save the results.

optional arguments:
  -h, --help            show this help message and exit
  --darknet DARKNET     Path to darknet executable file.
  --data-file DATA_FILE
                        Path to yolo *.data file.
  --cfg CFG             Path to the model *.cfg file.
  --weights WEIGHTS     Path to the model *.weights file.
  --width WIDTH         The model input width. Default = 512.
  --height HEIGHT       The model input height. Default = 512.
  --thresh THRESH       Confidence threshold. Default = 0.25.
  --test-file TEST_FILE
                        Path to yolo file with list of test data, usually "test.txt".
  --gt-path GT_PATH     Path to yolo folder with ground-truth data.
  --classes CLASSES     Path to yolo *.names file.
  --pascalvoc PASCALVOC
                        Path to pascalvoc.py script. Default = Object-Detection-Metrics/pascalvoc.py.
  --out OUT             Path to folder where save the results.
```

The script uses multiple steps:
1. Infers the darknet in test mode.
2. Converts ground-truth bounding boxes to "Object-Detection-Metrics" format.
3. Converts predicted bounding boxes to "Object-Detection-Metrics" format.
4. Removes all gt data that is not in the predicted set.
5. Runs pascalvoc.py