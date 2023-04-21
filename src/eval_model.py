import argparse
import subprocess
import shutil
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

DEFAULT_W = 512
DEFAULT_H = 512
DEFAULT_TH = 0.25
DEFAAULT_BETA_F = 1.0
DEFAULT_IOU_TH = 0.45
DEFAULT_PASCALVOC_PATH = os.path.join(DIR_PATH, os.pardir, "Object-Detection-Metrics", "pascalvoc.py")


def get_classes_and_test_file(ipath):
    classes_path = ''
    test_path = ''
    with open(ipath, 'rt') as f:
        for line in f.readlines():
            if 'valid' in line:
                test_path = line.split()[-1]
            elif 'names' in line:
                classes_path = line.split()[-1]

    return classes_path, test_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate yolov4 using mAP from 'Object-Detection-Metrics' project.")
    parser.add_argument(
        "--nn_shard_lite", required=True, help='Path to nn_shard_lite executable file.'
    )
    parser.add_argument(
        "--data-file", required=True, help='Path to yolo\'s *.data file.'
    )
    parser.add_argument(
        "--weights", required=True, help='Path to the model *.weights file.'
    )
    parser.add_argument(
        "--width", type=int, default=DEFAULT_W,
        help=f'The model\'s input width. Default = {DEFAULT_W}.'
    )
    parser.add_argument(
        "--height", type=int, default=DEFAULT_H,
        help=f'The model\'s input height. Default = {DEFAULT_H}.'
    )
    parser.add_argument(
        "--thresh", type=float, default=DEFAULT_TH,
        help=f'Confidence threshold. Default = {DEFAULT_TH}.'
    )
    parser.add_argument(
        "--iou-th", type=float, default=DEFAULT_IOU_TH,
        help=f'IoU threshold. Default = {DEFAULT_IOU_TH}.'
    )
    parser.add_argument(
        "--pascalvoc", default=DEFAULT_PASCALVOC_PATH,
        help=f'Path to pascalvoc.py script. Default = {DEFAULT_PASCALVOC_PATH}.'
    )
    parser.add_argument(
        "--out", required=True, help="Path to folder where save the results."
    )
    parser.add_argument(
        "--beta-f", type=float, default=DEFAAULT_BETA_F,
        help='Beta value for F_beta score, beta is chosen such that recall is '\
            f'considered beta times as important as precision. Default = {DEFAAULT_BETA_F}.'
    )
    args = parser.parse_args()
    
    classes, test_file = get_classes_and_test_file(args.data_file)

    tmp_gt = os.path.join(os.getcwd(), "tmp-gt")
    tmp_det = os.path.join(os.getcwd(), "tmp-det")

    with open("message.txt", 'w') as f:
        pass

    shutil.rmtree(args.out, ignore_errors=True)
    os.mkdir(args.out)
    
    env = os.environ.copy()
    env["INPATH"] = test_file
    env["WEIGHTS"] = args.weights
    env["MODEL_NAME"] = "yolov8"
    env["NAMES_PATH"] = classes
    env["IMG_W"] = str(args.width)
    env["IMG_H"] = str(args.height)
    env["OUTPATH"] = tmp_det
    env["CONF_TH"] = str(args.thresh)
    env["CALLBACKS"] = "file"
    env["IOU_TH"] = str(args.iou_th)

    print("\n    RUN MODEL\n")
    subprocess.run(["python3", args.nn_shard_lite], env=env)
    print("\n    CONVERT GT BBOXES\n")
    subprocess.run([
        "python3", os.path.join(DIR_PATH, "center2topleft.py"), "-i", test_file, "-o", tmp_gt, "-c", classes
    ])
    print("\n    REMOVE EXTRA GT FILES\n")
    subprocess.run([
        "python3", os.path.join(DIR_PATH, "rm_extra_gt_files.py"), "--gt", tmp_gt, "--det", tmp_det
    ])
    print("\n    RUN PASCALVOC\n")
    subprocess.run([
        "python3", args.pascalvoc, "-gt", tmp_gt, "-det", tmp_det, "-gtformat", "xywh", "-detformat", "xywh",
        "-gtcoords", "rel", "-detcoords", "rel", "-imgsize", f"{args.width},{args.height}", "-np", "-sp", args.out,
        "-b", str(args.beta_f)
    ])

    # shutil.rmtree(tmp_gt, ignore_errors=True)
    # shutil.rmtree(tmp_det, ignore_errors=True)
    # os.remove("message.txt")
