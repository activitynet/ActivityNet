import argparse

from eval_classification import ANETclassification

def main(ground_truth_filename, prediction_filename,
         subset='validation', verbose=True, check_status=True):
    anet_classification = ANETclassification(ground_truth_filename,
                                             prediction_filename,
                                             subset=subset, verbose=verbose,
                                             check_status=True)
    anet_classification.evaluate()

def parse_input():
    description = ('This script allows you to evaluate the ActivityNet '
                   'untrimmed video classification task which is intended to '
                   'evaluate the ability of algorithms to predict activities '
                   'in untrimmed video sequences.')
    p = argparse.ArgumentParser(description=description)
    p.add_argument('ground_truth_filename',
                   help='Full path to json file containing the ground truth.')
    p.add_argument('prediction_filename',
                   help='Full path to json file containing the predictions.')
    p.add_argument('--subset', default='validation',
                   help=('String indicating subset to evaluate: '
                         '(training, validation)'))
    p.add_argument('--verbose', type=bool, default=True)
    p.add_argument('--check_status', type=bool, default=True)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_input()
    main(**vars(args))
