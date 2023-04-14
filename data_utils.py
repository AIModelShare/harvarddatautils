import glob
import os

import mat73
import numpy as np
import pandas as pd
import persist_to_disk as ptd
import scipy.stats
import tqdm
from scipy.signal import butter, filtfilt, iirnotch
from torch.utils.data import Dataset

DATA_PATH = '/sbgenomics/project-files/Columbia/Training-set'
TEST_DATA_PATH = '/sbgenomics/project-files/Columbia/Test-set'
SAMPLING_RATE = 200
CLASSES = ['Other', 'Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA']

TRAIN = 'train'
VALID = 'val'
TEST = 'test'

def channel_transform(X):
    # to bi-polar signals
    temp = np.zeros_like(X)
    temp[0] = X[0] - X[4]
    temp[1] = X[4] - X[5]
    temp[2] = X[5] - X[6]
    temp[3] = X[6] - X[7]
    temp[4] = X[11] - X[15]
    temp[5] = X[15] - X[16]
    temp[6] = X[16] - X[17]
    temp[7] = X[17] - X[18]
    temp[8] = X[0] - X[1]
    temp[9] = X[1] - X[2]
    temp[10] = X[2] - X[3]
    temp[11] = X[3] - X[7]
    temp[12] = X[11] - X[12]
    temp[13] = X[12] - X[13]
    temp[14] = X[13] - X[14]
    temp[15] = X[14] - X[18]
    return temp[:16]#.astype("int16")

def denoise_channel(ts, bandpass, notch_freq, signal_freq):
    """
    bandpass: (low, high)
    """
    nyquist_freq = 0.5 * signal_freq
    filter_order = 2

    low = bandpass[0] / nyquist_freq
    high = bandpass[1] / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    ts_out = filtfilt(b, a, ts)

    quality_factor = 30.0
    b_notch, a_notch = iirnotch(notch_freq, quality_factor, signal_freq)
    ts_out = filtfilt(b_notch, a_notch, ts_out)

    return np.array(ts_out)

# This decorator is used to cache the data on disk.
# Please refer to the documentation of persist_to_disk on pip, or remove this if you don't need it.
@ptd.persistf(hashsize=10000)
def _read_and_transform_x(data_path, sampling_rate=SAMPLING_RATE):
    """This function reads the data from the mat file and transforms it.
    You could customize your own transforms instead of using this sequence.
    """
    x = mat73.loadmat(data_path)['data_50sec']
    # Step 1: Take the middle 10 seconds out of 50 seconds
    x = x[:, sampling_rate * (20): sampling_rate * (30)]
    # Step 2: transform to bi-polar signals
    x = channel_transform(x)
    # Step 3: perform bandpass and notch filter
    x = denoise_channel(x, [0.5, 40.0], 60.0, sampling_rate)
    return x

@ptd.persistf()
def _read_labels(data_path):
    data_dict = mat73.loadmat(data_path)
    return {'subject_ID': data_dict['subject_ID'], 'votes': data_dict['votes'],
           'path': data_path, 'key': os.path.basename(data_path).split('.')[0]}

def get_split_indices(seed, split_ratio, n, names=None):
    """Compute the split indices for a given seed and split ratio.
    """
    if names is None:
        names = [TRAIN, VALID, TEST]
    assert len(split_ratio) in {2,3}
    perm = np.random.RandomState(seed).permutation(n)
    split_ratio = np.asarray(split_ratio).cumsum() / sum(split_ratio)
    cuts = [int(_s* n) for _s in split_ratio]
    return {
        names[i]: perm[cuts[i-1]:cuts[i]] if i > 0 else perm[:cuts[0]]
        for i in range(len(split_ratio))
        }

class ColumbiaData(Dataset):
    DATASET = 'IIIC'
    CLASSES = CLASSES
    LABEL_MAP = {_n:_i for _i, _n in enumerate(CLASSES)}
    def __init__(self, split, split_ratio=[0.6, 0.2, 0.2], seed=42, debug=True, data_dir=DATA_PATH):
        PID_COL = 'subject_ID'
        super(ColumbiaData, self).__init__()

        _all = glob.glob(f"{data_dir}/*.mat")
        if debug:
            _all = [f for i, f in enumerate(_all) if i % 50 == 0]
        print("Reading labels...")
        _all = [_read_labels(_path) for _path in tqdm.tqdm(_all)]
        self._infos = pd.DataFrame([_['votes'].astype(int) for _ in _all])
        self._infos['majority'] = np.argmax(self._infos.values, axis=1)
        for _col in ['subject_ID', 'key', 'path']:
            self._infos[_col] = [_[_col] for _ in _all]



        # Create and pick the corresponding split
        if split is not None:
            print("Splitting Patients...")
            pids = sorted(self._infos[PID_COL].unique())
            pid_indices = get_split_indices(seed, split_ratio, len(pids))[split]
            pids = [pids[i] for i in pid_indices]
            self._infos = self._infos[self._infos[PID_COL].isin(pids)]
        print("Reading signals...")
        self.X = {row['key']: _read_and_transform_x(row['path']) for idx, row in tqdm.tqdm(self._infos.iterrows(), total=len(self._infos))}

        self.majority_only = True

    def _normalized(self, x, norm=2.5):
        # This is sample-wise rescale.
        # Recording-wise normalization might improve results.
        lb = np.percentile(x, norm)
        ub = np.percentile(x, 100-norm)
        x = x / np.clip(ub - lb, 1e-3, None)
        return x

    def __len__(self):
        return len(self._infos)

    def __getitem__(self, idx):
        record = self._infos.iloc[idx]
        key = record['key']
        X = self._normalized(self.X[key])
        target = record['majority']
        if not self.majority_only:
            V = record.reindex(columns=range(len(self.CLASSES))).values.astype(float)
            entropy = scipy.stats.entropy(V)
            target = np.asarray([target, entropy] + list(V))
        return {'data': X, 'target': target, 'index': key}

if __name__ == '__main__':
    # cache the data in parallel
    from multiprocessing import Pool
    def read_both(f):
        _read_labels(f)
        _read_and_transform_x(f)

    # cache train data
    tasks = glob.glob(f"{DATA_PATH}/*.mat")
    with Pool(16) as pool:
        for _ in tqdm.tqdm(pool.imap(read_both, tasks), total=len(tasks)):
            pass

    # cache test data
    test_tasks = glob.glob(f"{TEST_DATA_PATH}/*.mat")
    with Pool(16) as pool:
        for _ in tqdm.tqdm(pool.imap(read_both, test_tasks), total=len(test_tasks)):
            pass