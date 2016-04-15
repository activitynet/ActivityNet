import json
import urllib2

import numpy as np
import pandas as pd

from utils import get_blocked_videos
from utils import interpolated_prec_rec

class ANETclassification(object):
    GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PREDICTION_FIELDS = ['results', 'version']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 subset='validation', verbose=False, top_k=3):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.top_k = top_k
        self.ap = None
        self.hit_at_k = None
        # Retrieve blocked videos from server.
        self.blocked_videos = get_blocked_videos()
        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print '[INIT] Loaded annotations from {} subset.'.format(subset)
            nr_gt = len(self.ground_truth)
            print '\tNumber of ground truth instances: {}'.format(nr_gt)
            nr_pred = len(self.prediction)
            print '\tNumber of predictions: {}'.format(nr_pred)

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        if not all([field in data.keys() for field in self.gt_fields]):
            raise IOError('Please input a valid ground truth file.')

        # Initialize data frame
        columns=['video-id', 'label']
        ground_truth = pd.DataFrame(columns=columns)
        activity_index, cidx = {}, 0
        idx = 0
        for videoid, v in data['database'].iteritems():
            if self.subset != v['subset']:
                continue
            if videoid in self.blocked_videos:
                continue
            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    cidx += 1
                ground_truth.loc[idx] = {'video-id': videoid,
                                         'label': activity_index[ann['label']]}
                idx += 1
        ground_truth = ground_truth.drop_duplicates().reset_index(drop=True)
        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')

        # Initialize data frame
        columns = ['video-id', 'label', 'score']
        prediction = pd.DataFrame(columns=columns)
        idx = 0
        for videoid, v in data['results'].iteritems():
            if videoid in self.blocked_videos:
                continue
            for result in v:
                label = self.activity_index[result['label']]
                prediction.loc[idx] = {'video-id': videoid,
                                       'label': label,
                                       'score': result['score']}
                idx += 1
        return prediction

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros(len(self.activity_index.items()))
        for activity, cidx in self.activity_index.iteritems():
            gt_idx = self.ground_truth['label'] == cidx
            pred_idx = self.prediction['label'] == cidx
            ap[cidx] = compute_average_precision_classification(
                self.ground_truth.loc[gt_idx].reset_index(drop=True),
                self.prediction.loc[pred_idx].reset_index(drop=True))
        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        ap = self.wrapper_compute_average_precision()
        hit_at_k = compute_video_hit_at_k(self.ground_truth,
                                          self.prediction, top_k=self.top_k)
        if self.verbose:
            print ('[RESULTS] Performance on ActivityNet untrimmed video '
                   'classification task.')
            print '\tMean Average Precision: {}'.format(ap.mean())
            print '\tHit@{}: {}'.format(self.top_k, hit_at_k)
        self.ap = ap
        self.hit_at_k = hit_at_k

################################################################################
# Metrics
################################################################################

def compute_average_precision_classification(ground_truth, prediction):
    """Compute average precision (classification task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matched as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 'score']

    Outputs
    -------
    ap : float
        Average precision score.
    """
    npos = float(len(ground_truth))
    lock_gt = np.ones(len(ground_truth)) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros(len(prediction))
    fp = np.zeros(len(prediction))

    # Assigning true positive to truly grount truth instances.
    for idx in range(len(prediction)):
        this_pred = prediction.loc[idx]
        gt_idx = ground_truth['video-id'] == this_pred['video-id']
        # Check if there is at least one ground truth in the video associated.
        if not gt_idx.any():
            fp[idx] = 1
            continue
        this_gt = ground_truth.loc[gt_idx].reset_index()
        if lock_gt[this_gt['index']] >= 0:
            fp[idx] = 1
        else:
            tp[idx] = 1
            lock_gt[this_gt['index']] = idx

    # Computing prec-rec
    tp = np.cumsum(tp).astype(np.float)
    fp = np.cumsum(fp).astype(np.float)
    rec = tp / npos
    prec = tp / (tp + fp)
    return interpolated_prec_rec(rec, prec)

def compute_video_hit_at_k(ground_truth, prediction, top_k=3):
    """Compute accuracy at k prediction between ground truth and
    predictions data frames. This code is greatly inspired by evaluation
    performed in Karpathy et al. CVPR14.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 'label']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 'label', 'score']

    Outputs
    -------
    acc : float
        Top k accuracy score.
    """
    video_ids = np.unique(ground_truth['video-id'].values)
    n_videos = float(video_ids.size)
    hits = 0
    for vid in video_ids:
        pred_idx = prediction['video-id'] == vid
        if not pred_idx.any():
            continue
        this_pred = prediction.loc[pred_idx].reset_index(drop=True)
        # Get top K predictions sorted by decreasing score.
        sort_idx = this_pred['score'].values.argsort()[::-1][:top_k]
        this_pred = this_pred.loc[sort_idx].reset_index(drop=True)
        # Get labels and compare against ground truth.
        pred_label = this_pred['label'].tolist()
        gt_idx = ground_truth['video-id'] == vid
        gt_label = ground_truth.loc[gt_idx]['label'].tolist()
        if any([this_label in pred_label for this_label in gt_label]):
            hits += 1
    return hits / n_videos
