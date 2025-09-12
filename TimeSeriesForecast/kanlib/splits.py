import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def split_by_file(file_ids, ratios=(0.6,0.2,0.2), seed=42):
    files = np.array(sorted(set(file_ids)))
    train_f, rest = train_test_split(files, train_size=ratios[0], random_state=seed)
    val_size = ratios[1]/(1.0 - ratios[0])
    val_f, test_f = train_test_split(rest, train_size=val_size, random_state=seed)
    return set(train_f), set(val_f), set(test_f)

def loso_split(subject_ids):
    subs = sorted(set(subject_ids))
    for s in subs:
        train = [x for x in subs if x != s]
        yield set(train), {s}
