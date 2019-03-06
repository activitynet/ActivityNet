r"""Compute action detection performance for the AVA dataset.

Please send any questions about this code to the Google Group ava-dataset-users:
https://groups.google.com/forum/#!forum/ava-dataset-users

Example usage:
python -O get_ava_performance.py \
  -l ava/ava_action_list_v2.1_for_activitynet_2018.pbtxt.txt \
  -g ava_val_v2.1.csv \
  -e ava_val_excluded_timestamps_v2.1.csv \
  -d your_results.csv
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import defaultdict
import csv
import heapq
import logging
import pprint
import sys
import time
import numpy as np

from ava import object_detection_evaluation
from ava import standard_fields


def print_time(message, start):
  logging.info("==> %g seconds to %s", time.time() - start, message)


def make_image_key(video_id, timestamp):
  """Returns a unique identifier for a video id & timestamp."""
  return "%s,%04d" % (video_id, int(timestamp))


def read_csv(csv_file, class_whitelist=None, capacity=0):
  """Loads boxes and class labels from a CSV file in the AVA format.

  CSV file format described at https://research.google.com/ava/download.html.

  Args:
    csv_file: A file object.
    class_whitelist: If provided, boxes corresponding to (integer) class labels
      not in this set are skipped.
    capacity: Maximum number of labeled boxes allowed for each example.
      Default is 0 where there is no limit.

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
  entries = defaultdict(list)
  boxes = defaultdict(list)
  labels = defaultdict(list)
  scores = defaultdict(list)
  reader = csv.reader(csv_file)
  for row in reader:
    assert len(row) in [7, 8], "Wrong number of columns: " + row
    image_key = make_image_key(row[0], row[1])
    x1, y1, x2, y2 = [float(n) for n in row[2:6]]
    action_id = int(row[6])
    if class_whitelist and action_id not in class_whitelist:
      continue
    score = 1.0
    if len(row) == 8:
      score = float(row[7])
    if capacity < 1 or len(entries[image_key]) < capacity:
      heapq.heappush(entries[image_key],
                     (score, action_id, y1, x1, y2, x2))
    elif score > entries[image_key][0][0]:
      heapq.heapreplace(entries[image_key],
                        (score, action_id, y1, x1, y2, x2))
  for image_key in entries:
    # Evaluation API assumes boxes with descending scores
    entry = sorted(entries[image_key], key=lambda tup: -tup[0])
    for item in entry:
      score, action_id, y1, x1, y2, x2 = item
      boxes[image_key].append([y1, x1, y2, x2])
      labels[image_key].append(action_id)
      scores[image_key].append(score)
  print_time("read file " + csv_file.name, start)
  return boxes, labels, scores


def read_exclusions(exclusions_file):
  """Reads a CSV file of excluded timestamps.

  Args:
    exclusions_file: A file object containing a csv of video-id,timestamp.

  Returns:
    A set of strings containing excluded image keys, e.g. "aaaaaaaaaaa,0904",
    or an empty set if exclusions file is None.
  """
  excluded = set()
  if exclusions_file:
    reader = csv.reader(exclusions_file)
    for row in reader:
      assert len(row) == 2, "Expected only 2 columns, got: " + row
      excluded.add(make_image_key(row[0], row[1]))
  return excluded


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
    elif line.startswith("  id:") or line.startswith("  label_id:"):
      class_id = int(line.strip().split(" ")[-1])
      labelmap.append({"id": class_id, "name": name})
      class_ids.add(class_id)
  return labelmap, class_ids


def run_evaluation(labelmap, groundtruth, detections, exclusions):
  """Runs evaluations given input files.

  Args:
    labelmap: file object containing map of labels to consider, in pbtxt format
    groundtruth: file object
    detections: file object
    exclusions: file object or None.
  """
  categories, class_whitelist = read_labelmap(labelmap)
  logging.info("CATEGORIES (%d):\n%s", len(categories),
               pprint.pformat(categories, indent=2))
  excluded_keys = read_exclusions(exclusions)

  pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(
      categories)

  # Reads the ground truth data.
  boxes, labels, _ = read_csv(groundtruth, class_whitelist, 0)
  start = time.time()
  for image_key in boxes:
    if image_key in excluded_keys:
      logging.info(("Found excluded timestamp in ground truth: %s. "
                    "It will be ignored."), image_key)
      continue
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
  boxes, labels, scores = read_csv(detections, class_whitelist, 50)
  start = time.time()
  for image_key in boxes:
    if image_key in excluded_keys:
      logging.info(("Found excluded timestamp in detections: %s. "
                    "It will be ignored."), image_key)
      continue
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
  parser.add_argument(
      "-e",
      "--exclusions",
      help=("Optional CSV file containing videoid,timestamp pairs to exclude "
            "from evaluation."),
      type=argparse.FileType("r"),
      required=False)
  return parser.parse_args()


def main():
  logging.basicConfig(level=logging.INFO)
  args = parse_arguments()
  run_evaluation(**vars(args))


if __name__ == "__main__":
  main()
