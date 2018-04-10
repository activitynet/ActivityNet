r"""Compute action detection performance for the AVA dataset.

Example usage:
python -O get_ava_performance.py \
  -l ava/ava_action_list_v2.1_for_activitynet_2018.pbtxt.txt \
  -g ava_val_v2.1.csv \
  -d your_results.csv
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import defaultdict
import csv
import pprint
import sys
import time
import numpy as np

from ava import object_detection_evaluation
from ava import standard_fields


def print_time(message, start):
  print(
      "==> %g seconds to %s" % (time.time() - start, message), file=sys.stderr)


def read_csv(csv_file, class_whitelist=None):
  """Loads boxes and class labels from a CSV file in the AVA format.

  CSV file format described at https://research.google.com/ava/download.html.

  Args:
    csv_file: A file object.
    class_whitelist: If provided, boxes corresponding to (integer) class labels
      not in this set are skipped.

  Returns:
    boxes: A dictionary mapping each unique image key (string) to a list of
      boxes, given as coordinates [y1, x1, y2, x2].
    labels: A dictionary mapping each unique image key (string) to a list of
      integer class lables, matching the corresponding box in `boxes`.
    scores: A dictionary mapping each unique image key (string) to a list of
      score values lables, matching the corresponding label in `labels`. If
      scores are not provided in the csv, then they will default to 1.0.
  """
  start = time.time()
  boxes = defaultdict(list)
  labels = defaultdict(list)
  scores = defaultdict(list)
  reader = csv.reader(csv_file)
  for row in reader:
    assert len(row) in [7, 8], "Wrong number of columns: " + row
    video_id = row[0]
    timestamp = int(row[1])
    x1, y1, x2, y2 = [float(n) for n in row[2:6]]
    action_id = int(row[6])
    if class_whitelist and action_id not in class_whitelist:
      continue
    score = 1.0
    if len(row) == 8:
      score = float(row[7])
    image_key = "%s,%04d" % (video_id, timestamp)
    boxes[image_key].append([y1, x1, y2, x2])
    labels[image_key].append(action_id)
    scores[image_key].append(score)
  print_time("read file " + csv_file.name, start)
  return boxes, labels, scores


def read_labelmap(labelmap_file):
  """Reads a labelmap without the dependency on protocol buffers.

  Args:
    labelmap_file: A file object containing a label map protocol buffer.

  Returns:
    labelmap: The label map in the form used by the object_detection_evaluation
      module - a list of {"id": integer, "name": classname } dicts.
    class_ids: A set containing all of the valid class id integers.
  """
  labelmap = []
  class_ids = set()
  name = ""
  class_id = ""
  for line in labelmap_file:
    if line.startswith("  name:"):
      name = line.split('"')[1]
    elif line.startswith("  id:"):
      class_id = int(line.strip().split(" ")[-1])
      labelmap.append({"id": class_id, "name": name})
      class_ids.add(class_id)
  return labelmap, class_ids


def run_evaluation(labelmap, groundtruth, detections):
  """Runs evaluations given input files.

  Args:
    labelmap: file object containing map of labels to consider, in pbtxt format
    groundtruth: file object
    detections: file object
  """
  categories, class_whitelist = read_labelmap(labelmap)
  print(
      "CATEGORIES (%d):\n%s" % (len(categories),
                                pprint.pformat(categories, indent=2)),
      file=sys.stderr)

  pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(
      categories)

  # Reads the ground truth data.
  boxes, labels, _ = read_csv(groundtruth, class_whitelist)
  start = time.time()
  for image_key in boxes:
    pascal_evaluator.add_single_ground_truth_image_info(
        image_key, {
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array(boxes[image_key], dtype=float),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array(labels[image_key], dtype=int),
            standard_fields.InputDataFields.groundtruth_difficult:
                np.zeros(len(boxes[image_key]), dtype=bool)
        })
  print_time("convert groundtruth", start)

  # Reads detections data.
  boxes, labels, scores = read_csv(detections, class_whitelist)
  start = time.time()
  for image_key in boxes:
    pascal_evaluator.add_single_detected_image_info(
        image_key, {
            standard_fields.DetectionResultFields.detection_boxes:
                np.array(boxes[image_key], dtype=float),
            standard_fields.DetectionResultFields.detection_classes:
                np.array(labels[image_key], dtype=int),
            standard_fields.DetectionResultFields.detection_scores:
                np.array(scores[image_key], dtype=float)
        })
  print_time("convert detections", start)

  start = time.time()
  metrics = pascal_evaluator.evaluate()
  print_time("run_evaluator", start)
  pprint.pprint(metrics, indent=2)


def parse_arguments():
  """Parses command-line flags.

  Returns:
    args: a named tuple containing three file objects args.labelmap,
    args.groundtruth, and args.detections.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-l",
      "--labelmap",
      help="Filename of label map",
      type=argparse.FileType("r"),
      default="ava/ava_action_list_v2.1_for_activitynet_2018.pbtxt.txt")
  parser.add_argument(
      "-g",
      "--groundtruth",
      help="CSV file containing ground truth.",
      type=argparse.FileType("r"),
      required=True)
  parser.add_argument(
      "-d",
      "--detections",
      help="CSV file containing inferred action detections.",
      type=argparse.FileType("r"),
      required=True)
  return parser.parse_args()


def main():
  args = parse_arguments()
  run_evaluation(**vars(args))


if __name__ == "__main__":
  main()
