# YOLOv4 evaluation

The project evaluates darknet yolov4 using mAP metric from [Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics.git).

commit dcb285e7dea7e73d9480937d58de0e9bdfc20051

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
ython3 src/eval_model.py -h
usage: Evaluate yolov4 using mAP from 'Object-Detection-Metrics' project. 
  [-h] 
  --nn_shard_lite NN_SHARD_LITE 
  --data-file DATA_FILE 
  --weights WEIGHTS 
  [--width WIDTH] 
  [--height HEIGHT]
  [--thresh THRESH]
  [--pascalvoc PASCALVOC]
  --out OUT
  [--beta-f BETA_F]

optional arguments:
  -h, --help            show this help message and exit
  --nn_shard_lite NN_SHARD_LITE
                        Path to nn_shard_lite executable file.
  --data-file DATA_FILE
                        Path to yolo's *.data file.
  --weights WEIGHTS     Path to the model *.weights file.
  --width WIDTH         The model's input width. Default = 512.
  --height HEIGHT       The model's input height. Default = 512.
  --thresh THRESH       Confidence threshold. Default = 0.25.
  --pascalvoc PASCALVOC
                        Path to pascalvoc.py script. Default = /home/kyamkovyi/projects/drones/code/yolov4-evaluation/src/../Object-Detection-Metrics/pascalvoc.py.
  --out OUT             Path to folder where save the results.
  --beta-f BETA_F       Beta value for F_beta score, beta is chosen such that recall is considered beta times as important as precision. Default = 1.0.
```

The script uses multiple steps:
1. Infers the darknet in test mode.
2. Converts ground-truth bounding boxes to "Object-Detection-Metrics" format.
3. Converts predicted bounding boxes to "Object-Detection-Metrics" format.
4. Removes all gt data that is not in the predicted set.
5. Runs pascalvoc.py