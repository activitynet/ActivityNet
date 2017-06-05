import pandas as pd
import json
import argparse


status_and_reason_to_message_dict = {
    ('Downloaded', ''): ['Downloaded', 
                         'Exists'],

    ('Network',''): ['unable to download video data',
                     'The read operation timed out',
                     'Did not get any data blocks',
                     'giving up after 10 retries',
                     'Network is unreachable',
                     'content too short'],

    ('Unavailable','User-Removed'): ['This video is no longer available because the uploader has closed their YouTube account.',
                                     'account associated with this video has been terminated',
                                     'This video has been removed by the user.',
                                     'This video is not available.',
                                     'This video does not exist.'],

    ('Unavailable','Copyright'): ['multiple third-party notifications of copyright infringement.',
                                  'This video is no longer available due to a copyright claim',
                                  'blocked it on copyright grounds',
                                  'a duplicate of a previously uploaded video'],

    ('Unavailable','Country-Block'): ['The uploader has not made this video available in your country.',
                                      'who has blocked it in your country on copyright grounds.'],

    ('Unavailable','Spam'): ['policy on spam, deceptive practices, and scams.'],
    ('Unavailable','Nudity'): ['policy on nudity or sexual content.'],
    ('Unavailable','Sign-In'): ['Please sign in to view this video.'],
    ('Unavailable','Private'): ['This video is private.'],
    ('Unavailable','Guidelines'): ['Community Guidelines.'],
    ('Unavailable','Harassment and Bullying'): ['policy on harassment and bullying.'],
    ('Unavailable','Service-Terms'): ['Terms of Service.'],
    ('Unavailable','Harmful'): ['policy on harmful or dangerous content'],
    }

def get_status_and_reason(msg):
    for s_r, lst in status_and_reason_to_message_dict.iteritems():
        if any([x in msg for x in lst]):
            return s_r

    print("<get_status_and_reason>: error message is not matched with a status and a reason. message:", msg)

    return ('Other', msg)

def process_download_report(report):
    output = []
    for r in report:
        name, b, msg = r[0], r[1], r[2]
        output += [(name, get_status_and_reason(msg))]
    return output


def wrapper_process_download_reports(json_files):
    all_ouputs = []
    for f in json_files:
        with open(f, 'r') as fobj:
            report = json.load(fobj)
        all_ouputs += process_download_report(report)
    return all_ouputs

def main(input_csv, input_json, output_file, trim_format='%06d', num_input=1):
    json_files = []
    if num_input <= 1:
        json_files += [input_json]
    else:
        for i in range(num_input):
            json_files +=[input_json + ("-%02d" % (i+1))]
    
    all_ouputs = wrapper_process_download_reports(json_files)
    all_ouputs = dict(all_ouputs)

    dataset = pd.read_csv(input_csv)

    status_lst = []
    reason_lst = []
    for indx, row in dataset.iterrows():
        name = '%s_%s_%s' % (row['youtube_id'],
                             trim_format % row['time_start'],
                             trim_format % row['time_end'])

        s, r = all_ouputs[name]
        status_lst += [s]
        reason_lst += [r]
        if indx % 10000 == 0:
            print(indx)
    print("Done!!")
    dataset["status"] = status_lst
    dataset["reason"] = reason_lst
    
    dataset.to_csv(output_file, index=False)

if __name__ == '__main__':
    description = 'Helper script for processing the reports from downloading and trimming kinetics videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('input_csv', type=str,
                   help=('CSV file containing the following format: ' 
                         'label,   youtube_id,  time_start,  time_end,    split,   is_cc'))
    p.add_argument('input_json', type=str,
                   help=('base name for download report json files'),
                   default='download_report.json')
    p.add_argument('output_file', type=str,
                   help='Output csv file with statuses and reasons.')
    p.add_argument('-f', '--trim-format', type=str, default='%06d',
                   help=('This will be the format for the '
                         'filename of trimmed videos: '
                         'videoid_%0xd(start_time)_%0xd(end_time).mp4'))
    p.add_argument('-n', '--num_input', 
                    help=('number of input json files with the same base name input_json.'),
                    type=int, default=1)
    main(**vars(p.parse_args()))
