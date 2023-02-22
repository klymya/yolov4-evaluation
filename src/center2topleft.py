import argparse
import json
import glob
import os
import shutil


def center2topleft(x, y, w, h):
    return x - w / 2, y - h / 2


def convert_gt_line(line, classes):
    vals = [float(i) for i in line.split()]
    vals[0] = classes[int(vals[0])]
    vals[1: 3] = center2topleft(*vals[1:])
    vals = [str(i) for i in vals]
    return " ".join(vals) + '\n'


def convert_gt_file(inpath, outpath, classes):
    with open(inpath, 'rt') as f:
        gt_lines = f.readlines()

    gt_lines = [convert_gt_line(line, classes) for line in gt_lines]

    with open(outpath, 'wt') as f:
        f.writelines(gt_lines)


def convert_gt_files(inpath, outpath, classes):
    with open(inpath, 'rt') as f:
        inpaths = f.readlines()
    inpaths = [i.split('.')[0] + '.txt' for i in inpaths]

    for i in inpaths:
        tmp_outpath = f"{outpath}/{i.split('/')[-1]}"
        convert_gt_file(i, tmp_outpath, classes)


def convert_det_obj(det_obj):
    left, top = center2topleft(*[i for i in det_obj['relative_coordinates'].values()])
    return f"{det_obj['name']} {det_obj['confidence']} {left} {top} " \
        f"{det_obj['relative_coordinates']['width']} {det_obj['relative_coordinates']['height']}\n"


def convert_det_file(inpath, outpath):
    with open(inpath, 'rt') as f:
        det_result = json.load(f, strict=False)

    for det_objs in det_result:
        filename = det_objs['filename'].split('/')[-1].split('.')[0]
        det_lines = [convert_det_obj(obj) for obj in det_objs['objects']]

        with open(f"{outpath}/{filename}.txt", 'wt') as f:
            f.writelines(det_lines)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        "Convert bounding boxes [from center_x, center_y, width, height] to [left, top, width, height].\n" \
        "Can process GT and prediction files."
    )
    argparser.add_argument(
        '-i', help='path to bboxes. It could be folder with GT files or JSON file with yolov4 predictions.')
    argparser.add_argument('-o', help='path to save converted bboxes.')
    argparser.add_argument('-c', required=False, help='path to file with classes names. Only required for GT files.')

    args = argparser.parse_args()

    if os.path.isdir(args.o):
        shutil.rmtree(args.o, ignore_errors=True)
    os.mkdir(args.o)

    if args.i.endswith('.json'):
        convert_det_file(args.i, args.o)
    else:
        if args.c is None:
            raise ValueError("Path to classes names is required.")

        with open(args.c, 'rt') as f:
            classes = f.readlines()
        classes = [cl[:-1] for cl in classes]

        convert_gt_files(args.i, args.o, classes)
