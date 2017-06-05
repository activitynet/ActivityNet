# Kinetics - Downloader

## Usage
First, clone this repository and make sure that all the submodules are also cloned properly.
```
git clone https://github.com/activitynet/ActivityNet.git
cd ActivityNet/Crawler/Kinetics
```

Next, setup your environment
```
conda env create -f environment.yml
source activate kinetics
pip install --upgrade youtube-dl
```

Finally, download a dataset split by calling:
```
mkdir <data_dir>; python download.py {dataset_split}.csv <data_dir>
```
