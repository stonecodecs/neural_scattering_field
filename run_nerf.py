#!/usr/bin/env python3
import argparse

# NOTE: as of right now, can only use through demo.ipynb

## TODO: command line arguments interface
parser = argparse.ArgumentParser(
    description="""
    Multi-scatter NeRF. 
    (Input: a collection of images, Output: rendering of selected test pose)""")

parser.add_argument("input_path", type=str,
    help="The directory holding the dataset. "
    "Should have transforms_train.json and transforms_test.json,"
    "along with 'train' and 'test' directories to the images.")
parser.add_argument("output_path", type=str, help="Where to store outputs (renders for each test image selected).")
parser.add_argument("output_path", type=str, help="Environment Map lighting conditions for render.")
parser.add_argument("--render", nargs="*", type=str, default=None, help="Which test poses from transforms_test.json to render. If none, then uses all.")
parser.add_argument("--ckpt", type=str, default=None, help="If using an existing model, this is the location of model checkpoint to use.")
parser.add_argument("--build_only", action="store_false", help="No renders/testing, only trains the model. Overrides '--render' flag.")
parser.add_argument("--camera_zpos", action="store_false", help="If camera poses look toward the +z axis. Default is False (-z).")
parser.add_argument("--verbose", action="store_true", help="Prints intermediate training loss logs")

args = parser.parse_args()

## send to model ## 

# train (if ckpt is not None, otherwise skip) and print losses (if verbose)
# should always print tqdm progress bar

# save model checkpoint to output path

# render (if build_only is false)