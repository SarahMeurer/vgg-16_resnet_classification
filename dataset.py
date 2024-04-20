import pandas as pd
import numpy as np
import wfdb
import ast

import tqdm
import pathlib


def load_data(df, data_folder):
    # Better to use pathlib instead of os and strings
    data_folder = pathlib.Path(data_folder)

    # Get information about one example file
    _, info = wfdb.rdsamp(str(data_folder / df.filename_lr[0]))

    # Initialize dataset
    num_examples = df.shape[0]
    num_samples = info['sig_len']
    num_channels = info['n_sig']
    data = np.empty([num_examples, num_samples, num_channels])

    # Fill numpy array
    for i, filename in enumerate(tqdm.tqdm(df.filename_lr)):
        x, _ = wfdb.rdsamp(str(data_folder / filename))
        data[i, ] = x

    return data

path = '/home/datasets/ptbxl'
path = pathlib.Path(path)
sampling_rate = 100

# Load and convert annotation data
metadata = pd.read_csv(path / 'ptbxl_database.csv')
metadata.scp_codes = metadata.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load scp_statements.csv for diagnostic aggregation
meta_scp = pd.read_csv(path / 'scp_statements.csv', index_col=0)
meta_scp = meta_scp[meta_scp.diagnostic == 1]

# Define diagnostic classes
super_classes = ['NORM','STTC','CD','MI','HYP']

subdiag_dict = dict(meta_scp.diagnostic_class) # Key = subclasses, item = superclasses

def simple_diagnostic(scp_codes):
    vec = np.zeros(len(super_classes), dtype='int')
    # vec = np.zeros(diag_names.size, dtype='int')
    for key, item in scp_codes.items(): # Key = code, item = probability
        if key in meta_scp.index:
            diag_class = subdiag_dict[key]
            if item >= 50:
                vec[super_classes.index(diag_class)] = 1
                # vec[diag_names == diag_class] = 1

    # No diagnostic class present
    if vec.sum() == 0:
        return '???'

    return vec


# Simplify diagnostic
metadata['diagnostic_superclass'] = metadata.scp_codes.apply(simple_diagnostic)
metadata = metadata.drop(np.where(metadata.diagnostic_superclass == '???')[0])

# Load labels
Y = metadata.diagnostic_superclass.values
Y = np.array(Y.tolist())    

# Load raw ECG data
X = load_data(metadata, path)

# The dataset has 10 possible "validation folds",
# Folds 1-8 for training, fold 9 for validation and fold 10 for test
val_fold = 9
test_fold = 10

# Test (fold 10)
X_test = X[metadata.strat_fold.values == test_fold]
y_test = Y[metadata.strat_fold.values == test_fold]

# Validation (fold 9)
X_val = X[metadata.strat_fold.values == val_fold]
y_val = Y[metadata.strat_fold.values == val_fold]

# Train (folds 1-8)
X_train = X[metadata.strat_fold.values < val_fold]
y_train = Y[metadata.strat_fold.values < val_fold]

# Save metadata
metadata.to_csv('metadata.csv', index=False)
meta_scp.to_csv('metadata_scp.csv', index=False)

# Prepare data for saving
data = dict(
    X_train=X_train.astype('float32'),
    y_train=y_train.astype('int8'),
    X_val=X_val.astype('float32'),
    y_val=y_val.astype('int8'),
    X_test=X_test.astype('float32'),
    y_test=y_test.astype('int8')
)

# Save data as compressed numpy binaries
np.savez('data_val.npz', **data)
