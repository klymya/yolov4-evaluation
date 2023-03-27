###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: Feb 12th 2021                                                 #
###########################################################################################

####################################################################################################
#                                                                                                  #
# THE CURRENT VERSION WAS UPDATED WITH A VISUAL INTERFACE, INCLUDING MORE METRICS AND SUPPORTING   #
# OTHER FILE FORMATS. PLEASE ACCESS IT ACCESSED AT:                                                #
#                                                                                                  #
# https://github.com/rafaelpadilla/review_object_detection_metrics                                 #
#                                                                                                  #
# @Article{electronics10030279,                                                                    #
#     author         = {Padilla, Rafael and Passos, Wesley L. and Dias, Thadeu L. B. and Netto,    #
#                       Sergio L. and da Silva, Eduardo A. B.},                                    #
#     title          = {A Comparative Analysis of Object Detection Metrics with a Companion        #
#                       Open-Source Toolkit},                                                      #
#     journal        = {Electronics},                                                              #
#     volume         = {10},                                                                       #
#     year           = {2021},                                                                     #
#     number         = {3},                                                                        #
#     article-number = {279},                                                                      #
#     url            = {https://www.mdpi.com/2079-9292/10/3/279},                                  #
#     issn           = {2079-9292},                                                                #
#     doi            = {10.3390/electronics10030279}, }                                            #
####################################################################################################

####################################################################################################
# If you use this project, please consider citing:                                                 #
#                                                                                                  #
# @INPROCEEDINGS {padillaCITE2020,                                                                 #
#    author    = {R. {Padilla} and S. L. {Netto} and E. A. B. {da Silva}},                         #
#    title     = {A Survey on Performance Metrics for Object-Detection Algorithms},                #
#    booktitle = {2020 International Conference on Systems, Signals and Image Processing (IWSSIP)},#
#    year      = {2020},                                                                           #
#    pages     = {237-242},}                                                                       #
#                                                                                                  #
# This work is published at: https://github.com/rafaelpadilla/Object-Detection-Metrics             #
####################################################################################################

import argparse
import glob
import os
import shutil
import sys

import numpy as np

import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import BBFormat


def array2str(x):
    return np.array2string(x, formatter={'float': lambda x: f'{x:.2f}'}, separator=', ', max_line_width=1e6)


def get_scores_by_conf_th(scores, confs, conf_ths=None):
    if conf_ths is None:
        conf_ths = np.append(np.arange(1, 0.00, -0.05), 0.01)
    best_scores = []
    for th in conf_ths:
        try:
            score = scores[confs >= th][-1]
        except IndexError:
            score = 0
        best_scores.append(score)
    
    return np.array(best_scores), conf_ths


# Validate formats
def ValidateFormats(argFormat, argName, errors):
    if argFormat == 'xywh':
        return BBFormat.XYWH
    elif argFormat == 'xyrb':
        return BBFormat.XYX2Y2
    elif argFormat is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        errors.append('argument %s: invalid value. It must be either \'xywh\' or \'xyrb\'' %
                      argName)


# Validate mandatory args
def ValidateMandatoryArgs(arg, argName, errors):
    if arg is None:
        errors.append('argument %s: required argument' % argName)
    else:
        return True


def ValidateImageSize(arg, argName, argInformed, errors):
    errorMsg = 'argument %s: required argument if %s is relative' % (argName, argInformed)
    ret = None
    if arg is None:
        errors.append(errorMsg)
    else:
        arg = arg.replace('(', '').replace(')', '')
        args = arg.split(',')
        if len(args) != 2:
            errors.append('%s. It must be in the format \'width,height\' (e.g. \'600,400\')' %
                          errorMsg)
        else:
            if not args[0].isdigit() or not args[1].isdigit():
                errors.append(
                    '%s. It must be in INdiaTEGER the format \'width,height\' (e.g. \'600,400\')' %
                    errorMsg)
            else:
                ret = (int(args[0]), int(args[1]))
    return ret


# Validate coordinate types
def ValidateCoordinatesTypes(arg, argName, errors):
    if arg == 'abs':
        return CoordinatesType.Absolute
    elif arg == 'rel':
        return CoordinatesType.Relative
    elif arg is None:
        return CoordinatesType.Absolute  # default when nothing is passed
    errors.append('argument %s: invalid value. It must be either \'rel\' or \'abs\'' % argName)


def ValidatePaths(arg, nameArg, errors):
    if arg is None:
        errors.append('argument %s: invalid directory' % nameArg)
    elif os.path.isdir(arg) is False and os.path.isdir(os.path.join(currentPath, arg)) is False:
        errors.append('argument %s: directory does not exist \'%s\'' % (nameArg, arg))
    # elif os.path.isdir(os.path.join(currentPath, arg)) is True:
    #     arg = os.path.join(currentPath, arg)
    else:
        arg = os.path.join(currentPath, arg)
    return arg


def getBoundingBoxes(directory,
                     isGT,
                     bbFormat,
                     coordType,
                     allBoundingBoxes=None,
                     allClasses=None,
                     imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read ground truths
    os.chdir(directory)
    files = glob.glob("*.txt")
    files.sort()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            if isGT:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                x = float(splitLine[1])
                y = float(splitLine[2])
                w = float(splitLine[3])
                h = float(splitLine[4])
                bb = BoundingBox(nameOfImage,
                                 idClass,
                                 x,
                                 y,
                                 w,
                                 h,
                                 coordType,
                                 imgSize,
                                 BBType.GroundTruth,
                                 format=bbFormat)
            else:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                confidence = float(splitLine[1])
                x = float(splitLine[2])
                y = float(splitLine[3])
                w = float(splitLine[4])
                h = float(splitLine[5])
                bb = BoundingBox(nameOfImage,
                                 idClass,
                                 x,
                                 y,
                                 w,
                                 h,
                                 coordType,
                                 imgSize,
                                 BBType.Detected,
                                 confidence,
                                 format=bbFormat)
            allBoundingBoxes.addBoundingBox(bb)
            if idClass not in allClasses:
                allClasses.append(idClass)
        fh1.close()
    return allBoundingBoxes, allClasses


# Get current path to set default folders
currentPath = os.path.dirname(os.path.abspath(__file__))

VERSION = '0.2 (beta)'

with open('message.txt', 'r') as f:
    message = f'\n\n{f.read()}\n\n'

print(message)

parser = argparse.ArgumentParser(
    prog='Object Detection Metrics - Pascal VOC',
    description=
    f'{message}\nThis project applies the most popular metrics used to evaluate object detection '
    'algorithms.\nThe current implemention runs the Pascal VOC metrics.\nFor further references, '
    'please check:\nhttps://github.com/rafaelpadilla/Object-Detection-Metrics',
    epilog="Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)")
parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
# Positional arguments
# Mandatory
parser.add_argument('-gt',
                    '--gtfolder',
                    dest='gtFolder',
                    default=os.path.join(currentPath, 'groundtruths'),
                    metavar='',
                    help='folder containing your ground truth bounding boxes')
parser.add_argument('-det',
                    '--detfolder',
                    dest='detFolder',
                    default=os.path.join(currentPath, 'detections'),
                    metavar='',
                    help='folder containing your detected bounding boxes')
# Optional
parser.add_argument('-t',
                    '--threshold',
                    dest='iouThreshold',
                    type=float,
                    default=0.5,
                    metavar='',
                    help='IOU threshold. Default 0.5')
parser.add_argument('-gtformat',
                    dest='gtFormat',
                    metavar='',
                    default='xywh',
                    help='format of the coordinates of the ground truth bounding boxes: '
                    '(\'xywh\': <left> <top> <width> <height>)'
                    ' or (\'xyrb\': <left> <top> <right> <bottom>)')
parser.add_argument('-detformat',
                    dest='detFormat',
                    metavar='',
                    default='xywh',
                    help='format of the coordinates of the detected bounding boxes '
                    '(\'xywh\': <left> <top> <width> <height>) '
                    'or (\'xyrb\': <left> <top> <right> <bottom>)')
parser.add_argument('-gtcoords',
                    dest='gtCoordinates',
                    default='abs',
                    metavar='',
                    help='reference of the ground truth bounding box coordinates: absolute '
                    'values (\'abs\') or relative to its image size (\'rel\')')
parser.add_argument('-detcoords',
                    default='abs',
                    dest='detCoordinates',
                    metavar='',
                    help='reference of the ground truth bounding box coordinates: '
                    'absolute values (\'abs\') or relative to its image size (\'rel\')')
parser.add_argument('-imgsize',
                    dest='imgSize',
                    metavar='',
                    help='image size. Required if -gtcoords or -detcoords are \'rel\'')
parser.add_argument('-sp',
                    '--savepath',
                    dest='savePath',
                    metavar='',
                    help='folder where the plots are saved')
parser.add_argument('-np',
                    '--noplot',
                    dest='showPlot',
                    action='store_false',
                    help='no plot is shown during execution')
parser.add_argument('-b',
                    '--beta-f',
                    dest='betaF',
                    type=float,
                    default=1.0,
                    metavar='',
                    help='beta for f_beta score')
args = parser.parse_args()

iouThreshold = args.iouThreshold

# Arguments validation
errors = []
# Validate formats
gtFormat = ValidateFormats(args.gtFormat, '-gtformat', errors)
detFormat = ValidateFormats(args.detFormat, '-detformat', errors)
# Groundtruth folder
if ValidateMandatoryArgs(args.gtFolder, '-gt/--gtfolder', errors):
    gtFolder = ValidatePaths(args.gtFolder, '-gt/--gtfolder', errors)
else:
    # errors.pop()
    gtFolder = os.path.join(currentPath, 'groundtruths')
    if os.path.isdir(gtFolder) is False:
        errors.append('folder %s not found' % gtFolder)
# Coordinates types
gtCoordType = ValidateCoordinatesTypes(args.gtCoordinates, '-gtCoordinates', errors)
detCoordType = ValidateCoordinatesTypes(args.detCoordinates, '-detCoordinates', errors)
betaF = args.betaF
imgSize = (0, 0)
if gtCoordType == CoordinatesType.Relative:  # Image size is required
    imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-gtCoordinates', errors)
if detCoordType == CoordinatesType.Relative:  # Image size is required
    imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-detCoordinates', errors)
# Detection folder
if ValidateMandatoryArgs(args.detFolder, '-det/--detfolder', errors):
    detFolder = ValidatePaths(args.detFolder, '-det/--detfolder', errors)
else:
    # errors.pop()
    detFolder = os.path.join(currentPath, 'detections')
    if os.path.isdir(detFolder) is False:
        errors.append('folder %s not found' % detFolder)
if args.savePath is not None:
    savePath = ValidatePaths(args.savePath, '-sp/--savepath', errors)
else:
    savePath = os.path.join(currentPath, 'results')
# Validate savePath
# If error, show error messages
if len(errors) != 0:
    print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                [-detformat] [-save]""")
    print('Object Detection Metrics: error(s): ')
    [print(e) for e in errors]
    sys.exit()

# Check if path to save results already exists and is not empty
if os.path.isdir(savePath) and os.listdir(savePath):
    key_pressed = ''
    while key_pressed.upper() not in ['Y', 'N']:
        print(f'Folder {savePath} already exists and may contain important results.\n')
        print(f'Enter \'Y\' to continue. WARNING: THIS WILL REMOVE ALL THE CONTENTS OF THE FOLDER!')
        print(f'Or enter \'N\' to abort and choose another folder to save the results.')
        key_pressed = input('')

    if key_pressed.upper() == 'N':
        print('Process canceled')
        sys.exit()

# Clear folder and save results
shutil.rmtree(savePath, ignore_errors=True)
os.makedirs(savePath)
# Show plot during execution
showPlot = args.showPlot

# print('iouThreshold= %f' % iouThreshold)
# print('savePath = %s' % savePath)
# print('gtFormat = %s' % gtFormat)
# print('detFormat = %s' % detFormat)
# print('gtFolder = %s' % gtFolder)
# print('detFolder = %s' % detFolder)
# print('gtCoordType = %s' % gtCoordType)
# print('detCoordType = %s' % detCoordType)
# print('showPlot %s' % showPlot)

# Get groundtruth boxes
allBoundingBoxes, allClasses = getBoundingBoxes(gtFolder,
                                                True,
                                                gtFormat,
                                                gtCoordType,
                                                imgSize=imgSize)
# Get detected boxes
allBoundingBoxes, allClasses = getBoundingBoxes(detFolder,
                                                False,
                                                detFormat,
                                                detCoordType,
                                                allBoundingBoxes,
                                                allClasses,
                                                imgSize=imgSize)
allClasses.sort()

evaluator = Evaluator()
acc_AP = 0
validClasses = 0
acc_total_positives = 0
acc_f1 = 0
acc_fbeta = 0
acc_weighted_f1 = 0
acc_weighted_fbeta = 0
acc_total_tp = 0
acc_total_fp = 0

f_scores_thresholded = []
precision_thresholded = []
recall_thresholded = []

# Plot Precision x Recall curve
detections = evaluator.PlotPrecisionRecallCurve(
    allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
    IOUThreshold=iouThreshold,  # IOU threshold
    method=MethodAveragePrecision.EveryPointInterpolation,
    showAP=True,  # Show Average Precision in the title of the plot
    showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
    savePath=savePath,
    showGraphic=showPlot,
    betaF=betaF)

f = open(os.path.join(savePath, 'results.txt'), 'w')
f.write('Object Detection Metrics\n')
f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
f.write('Average Precision (AP), Precision and Recall per class:')

f.write('\n\nbeta: %.2f' % betaF)

# each detection is a class
print(f"| {'class':^30} | {'AP':^10} | {'f1':^10} | {'f_beta':^10} |")
for metricsPerClass in detections:

    # Get metric values per each class
    cl = metricsPerClass['class']
    ap = metricsPerClass['AP']
    precision = metricsPerClass['precision']
    recall = metricsPerClass['recall']
    totalPositives = metricsPerClass['total positives']
    total_TP = metricsPerClass['total TP']
    total_FP = metricsPerClass['total FP']
    confidence = metricsPerClass['confidence']
    f1_score = metricsPerClass['f1_score']
    f_beta_score = metricsPerClass['f_beta_score']

    if totalPositives > 0:
        validClasses = validClasses + 1
        acc_AP = acc_AP + ap
        acc_f1 += f1_score[-1]
        acc_fbeta += f_beta_score[-1]
        
        acc_total_positives += totalPositives
        acc_total_tp += total_TP
        acc_total_fp += total_FP
        acc_weighted_f1 += f1_score[-1] * totalPositives
        acc_weighted_fbeta += f_beta_score[-1] * totalPositives
        
        best_scores, conf_ths = get_scores_by_conf_th(f_beta_score, confidence)
        best_precision, _ = get_scores_by_conf_th(precision, confidence)
        best_recall, _ = get_scores_by_conf_th(recall, confidence)
        f_scores_thresholded.append(best_scores)
        precision_thresholded.append(best_precision)
        recall_thresholded.append(best_recall)
        
        ap_str = "{0:.2f}%".format(ap * 100)
        f1_str = "{0:.2f}%".format(f1_score[-1] * 100)
        fbeta_str = "{0:.2f}%".format(f_beta_score[-1] * 100)
        print(f"| {cl:<30} | {ap_str:>10} | {f1_str:>10} | {fbeta_str:>10} |")
        f.write(f'\n\nClass: {cl}')
        
        f.write(f'\nAVERAGE')
        f.write(f'\nAP          : {ap_str}')
        f.write(f'\nF1 score    : {f1_str}')
        f.write(f'\nF_beta score: {fbeta_str}')
        
        f.write(f'\nGROUPED')
        f.write(f'\nConfidence ths: {array2str(conf_ths)}')
        f.write(f'\nF_beta score  : {array2str(best_scores)}')
        f.write(f'\nPrecision     : {array2str(best_precision)}')
        f.write(f'\nRecall        : {array2str(best_recall)}')

        f.write(f'\nALL DETECTIONS')
        f.write(f'\nPrecision     : {array2str(precision)}')
        f.write(f'\nRecall        : {array2str(recall)}')
        f.write(f'\nConfidence    : {array2str(confidence)}')
        f.write(f'\nF1 score      : {array2str(f1_score)}')
        f.write(f'\nF_beta score  : {array2str(f_beta_score)}')

mAP = acc_AP / validClasses

macro_f1 = acc_f1 / validClasses
macro_fbeta = acc_fbeta / validClasses

weighted_f1 = acc_weighted_f1 / acc_total_positives
weighted_fbeta = acc_weighted_fbeta / acc_total_positives

rec = acc_total_tp / acc_total_positives
prec = np.divide(acc_total_tp, (acc_total_fp + acc_total_tp))
f1_score = 2 * prec * rec / (prec + rec + 1e-6)
f_beta_score = (1 + betaF**2) * prec * rec / (betaF**2 * prec + rec + 1e-6)

mAP_str = "{0:.2f}%".format(mAP * 100)
macro_f1_str = "{0:.2f}%".format(macro_f1 * 100)
macro_fbeta_str = "{0:.2f}%".format(macro_fbeta * 100)
weighted_f1_str = "{0:.2f}%".format(weighted_f1 * 100)
weighted_fbeta_str = "{0:.2f}%".format(weighted_fbeta * 100)
global_f1_str = "{0:.2f}%".format(f1_score * 100)
global_fbeta_str = "{0:.2f}%".format(f_beta_score * 100)

f_scores_thresholded = np.mean(f_scores_thresholded, axis=0)
recall_thresholded = np.mean(recall_thresholded, axis=0)
precision_thresholded = np.mean(precision_thresholded, axis=0)
idxs = np.argsort(f_scores_thresholded)[::-1]

f_scores_thresholded = f_scores_thresholded[idxs]
recall_thresholded = recall_thresholded[idxs]
precision_thresholded = precision_thresholded[idxs]
conf_ths = conf_ths[idxs]

print()
print(f'{"mAP":<20}: {mAP_str:>10}')
print(f'{"macro avg f1":<20}: {macro_f1_str:>10}')
print(f'{"macro avg f_beta":<20}: {macro_fbeta_str:>10}')
print(f'{"weighted avg f1":<20}: {weighted_f1_str:>10}')
print(f'{"weighted avg f_beta":<20}: {weighted_fbeta_str:>10}')
print(f'{"global avg f1":<20}: {global_f1_str:>10}')
print(f'{"global avg f_beta":<20}: {global_fbeta_str:>10}')

f.write('\n\nSUMMARY')
f.write('\nmAP                : %s' % mAP_str)
f.write('\nmacro avg f1       : %s' % macro_f1_str)
f.write('\nmacro avg f_beta   : %s' % macro_fbeta_str)
f.write('\nweighted avg f1    : %s' % weighted_f1_str)
f.write('\nweighted avg f_beta: %s' % weighted_fbeta_str)
f.write('\nglobal avg f1      : %s' % global_f1_str)
f.write('\nglobal avg f_beta  : %s' % global_fbeta_str)
f.write(f'\n\nsorted confidence th          : {array2str(conf_ths)}')
f.write(f'\nsorted macro avg f_beta scores: {array2str(f_scores_thresholded)}')
f.write(f'\nsorted precision              : {array2str(precision_thresholded)}')
f.write(f'\nsorted recall                 : {array2str(recall_thresholded)}')
