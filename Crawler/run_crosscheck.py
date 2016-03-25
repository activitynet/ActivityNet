from argparse import ArgumentParser
import glob
import json
import os

def crosscheck_videos(video_path, ann_file):
    # Get existing videos
    existing_vids = glob.glob("%s/*.mp4" % video_path)
    for idx, vid in enumerate(existing_vids):
        basename = os.path.basename(vid).split(".mp4")[0]
        if len(basename) == 13:
            existing_vids[idx] = basename[2:]
        elif len(basename) == 11:
            existing_vids[idx] = basename
        else:
            raise RuntimeError("Unknown filename format: %s", vid)
    # Read an get video IDs from annotation file
    with open(ann_file, "r") as fobj:
        anet_v_1_0 = json.load(fobj)
    all_vids = anet_v_1_0["database"].keys()
    non_existing_videos = []
    for vid in all_vids:
        if vid in existing_vids:
            continue
        else:
            non_existing_videos.append(vid)
    return non_existing_videos

def main(video_path, ann_file, output_filename):
    non_existing_videos = crosscheck_videos(video_path, ann_file)
    filename = os.path.join(video_path, "v_%s.mp4")
    cmd_base = "youtube-dl -f best -f mp4 "
    cmd_base += '"https://www.youtube.com/watch?v=%s" '
    cmd_base += '-o "%s"' % filename
    with open(output_filename, "w") as fobj:
        for vid in non_existing_videos:
            cmd = cmd_base % (vid, vid)
            fobj.write("%s\n" % cmd)

if __name__ == "__main__":
    parser = ArgumentParser(description="Script to double check video content.")
    parser.add_argument("video_path", help="Where are located the videos? (Full path)")
    parser.add_argument("ann_file", help="Where is the annotation file?")
    parser.add_argument("output_filename", help="Output script location.")
    args = vars(parser.parse_args())
    main(**args)
