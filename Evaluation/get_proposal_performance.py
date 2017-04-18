import argparse
import numpy as np

from eval_proposal import ANETproposal

def main(ground_truth_filename, proposal_filename, max_avg_nr_proposals=100,
         tiou_thresholds=np.linspace(0.5, 0.95, 10),
         subset='validation', verbose=True, check_status=True):

    anet_proposal = ANETproposal(ground_truth_filename, proposal_filename,
                                 tiou_thresholds=tiou_thresholds,
                                 max_avg_nr_proposals=max_avg_nr_proposals,
                                 subset=subset, verbose=True, check_status=True)
    anet_proposal.evaluate()

def parse_input():
    description = ('This script allows you to evaluate the ActivityNet '
                   'proposal task which is intended to evaluate the ability '
                   'of algorithms to generate activity proposals that temporally '
                   'localize activities in untrimmed video sequences.')
    p = argparse.ArgumentParser(description=description)
    p.add_argument('ground_truth_filename',
                   help='Full path to json file containing the ground truth.')
    p.add_argument('proposal_filename',
                   help='Full path to json file containing the proposals.')
    p.add_argument('--subset', default='validation',
                   help=('String indicating subset to evaluate: '
                         '(training, validation)'))
    p.add_argument('--verbose', type=bool, default=True)
    p.add_argument('--check_status', type=bool, default=True)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_input()
    main(**vars(args))
