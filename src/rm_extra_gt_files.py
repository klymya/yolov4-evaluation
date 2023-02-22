import glob
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt")
    parser.add_argument("--det")

    args = parser.parse_args()

    gts = glob.glob(f"{args.gt}/*.txt")
    gts = [i.split('/')[-1] for i in gts]
    dets = glob.glob(f"{args.det}/*.txt")
    dets = [i.split('/')[-1] for i in dets]

    extras = set(gts).difference(dets)

    for extra in extras:
        os.remove(f"{args.gt}/{extra}")

    print(f"removed {len(extras)} files.")
