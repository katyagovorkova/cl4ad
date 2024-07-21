import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

NAME_MAPPINGS = {
    0:'W-Boson',
    1:'QCD',
    2:'Z_2',
    3:'tt',
    4:'leptoquark',
    5:'ato4l',
    6:'hChToTauNu',
    7:'hToTauTau'
}

TRAIN_TEST_VAL_MAP = {
    'x_train':1,
    'x_test':0.333,
    'x_val':0.333
}

def zscore_preprocess(
        input_array,
        train=False,
        scaling_file=None, 
        for_transformer=False
        ):
    '''
    Normalizes using zscore scaling along pT only ->  x' = (x - μ) / σ
    Assumes (μ, σ) constants determined by average across training batch
    '''
    # Loads input as tensor and (μ, σ) constants predetermined from training batch.
    if train:
        tensor = input_array.copy()
        mu = np.mean(tensor[:,:,0,:])
        sigma = np.std(tensor[:,:,0,:])
        np.savez(scaling_file, mu=mu, sigma=sigma)

        normalized_tensor = (tensor - mu) / sigma

    else:
        tensor = input_array.copy()
        scaling_const = np.load(scaling_file)
        normalized_tensor = (tensor - scaling_const['mu']) / scaling_const['sigma']

    # Masking so unrecorded data remains 0
    mask = np.not_equal(input_array, 0)
    mask = np.squeeze(mask, -1)

    # Outputs normalized pT while preserving original values for eta and phi
    outputs = np.concatenate([normalized_tensor[:,:,0,:], input_array[:,:,1,:], input_array[:,:,2,:]], axis=2)

    if for_transformer:
        return np.reshape(outputs * mask, (-1, 19, 3))
    return np.reshape(outputs * mask, (-1, 57))


class CLBackgroundDataset:
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_filename, labels_filename, preprocess=None, n_events=-1, divisions=[.25, .25, .25, .25]):
        'Initialization'
        self.data = np.load(data_filename, mmap_mode='r')
        self.labels = np.load(labels_filename, mmap_mode='r')
        self.n_events = n_events if n_events!=-1 else len( self.data[ next(iter(self.data)) ] )  # use specificed number of events, otherwise use all
        self.scaled_dataset = dict()  # output dataset
        for k in self.data.keys():  
            if 'x_' in k:  # get the indices used from x and augmented x for train/test/val
                self.scaled_dataset[f"{k.replace('x','ix')}"], self.scaled_dataset[f"{k.replace('x','ixa')}"] = \
                    self.division_indicies(self.data[k], self.labels[k.replace('x', 'background_ID')], divisions)

        if preprocess:
            self.preprocess(preprocess)
        else:
            self.scaled_dataset['x_train'] = self.data['x_train']
            self.scaled_dataset['x_test'] = self.data['x_test']
            self.scaled_dataset['x_val'] = self.data['x_val']


    def division_indicies(self, data, labels, divisions):
            ix = []  # indices of corresponding samples
            ixa = []  # indices of corresponding augmented samples
            # (data, augmented event, label) = x[ix], x[ixa], labels[ix]
            used = 0  # keep track of how many samples have been filled up already
            
            total_sample = 0  
            for label_category in range(len(divisions)-1, -1, -1):  # iterate in reverse order

                indices = np.where(labels==label_category)[0]  # indices of labels that == label_category
                
                if label_category == len(divisions)-1:  # keep track of allowed total samples by number of tt events
                    # (all of tt events should make up e.g. 20% of the data)
                    # COMMENT: TRY WITHOUT USING ALL OF LABEL 3 DATA TO GET BETTER GENERALIZATION
                    total_sample = int(len(indices) / divisions[label_category])  # take the floor such that label 3 events are strictly enough
                
                if label_category != (0):  # if not the first category, simply calculate and round
                    # calculate number of samples needed as specified by division proportions
                    label_sample_size = round(divisions[label_category] * total_sample) 
                    used += label_sample_size

                else:  # if first category, subtract to get the remaining samples
                    label_sample_size = total_sample - used

                # sample the indices with/without replacement
                indices = list(np.random.choice(indices, size=label_sample_size, replace=False))
                # add sampled indices to ix
                ix.extend(indices)

                # augmentation is another event with the same label. for simplicity take the next event
                loc_aug = np.concatenate((indices[1:], indices[0:1]))
                ixa.extend(loc_aug)

            ix, ixa = shuffle(ix, ixa, random_state=0)

            # return data[ix], location_augmented, labels[ix].reshape((-1,1))
            return ix, ixa

    def preprocess(self, scaling_filename):
        tf = True if args.for_transformer else False
        # Normalizes train and testing features by x' = (x - μ) / σ, where μ, σ are predetermined constants
        self.scaled_dataset['x_train'] = zscore_preprocess(self.data['x_train'], train=True, scaling_file=scaling_filename, for_transformer=tf)
        self.scaled_dataset['x_test'] = zscore_preprocess(self.data['x_test'], scaling_file=scaling_filename, for_transformer=tf)
        self.scaled_dataset['x_val'] = zscore_preprocess(self.data['x_val'], scaling_file=scaling_filename, for_transformer=tf)

    def save(self, filename):

        np.savez(filename,
            x_train=self.scaled_dataset['x_train'],
            ix_train=self.scaled_dataset['ix_train'],
            ixa_train=self.scaled_dataset['ixa_train'],
            labels_train=self.labels['background_ID_train'],
            x_test=self.scaled_dataset['x_test'],
            ix_test=self.scaled_dataset['ix_test'],
            ixa_test=self.scaled_dataset['ixa_test'],
            labels_test=self.labels['background_ID_test'],
            x_val=self.scaled_dataset['x_val'],
            ix_val=self.scaled_dataset['ix_val'],
            ixa_val=self.scaled_dataset['ixa_val'],
            labels_val=self.labels['background_ID_val'],
            )
        print(f'{filename} successfully saved')

    def report_specs(self):
        '''
        Reports file specs: keys, shape pairs. If divisions, also reports number of samples from each label represented
        in dataset
        '''
        print('File Specs:')
        print(self.scaled_dataset.keys())
        for k in self.scaled_dataset:
            if 'ix_' in k:
                labels = self.labels[k.replace('ix_', 'background_ID_')].copy()[self.scaled_dataset[k]]
                labels = labels.reshape((labels.shape[0],))
                label_counts = labels.astype(int)
                label_counts = np.bincount(label_counts)
                for label, count in enumerate(label_counts):
                    print(f"Label {label, NAME_MAPPINGS[label]}: {count} occurances")


class CLSignalDataset:
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_filename, preprocess=None):
        'Initialization'
        self.data = np.load(data_filename, mmap_mode='r')
        self.labels = self.create_labels()
        self.scaled_dataset = dict()
        for k in self.data.keys():
            idx = np.random.choice(self.data[k].shape[0], size=self.data[k].shape[0], replace=False)
            np.random.shuffle(idx)
            self.scaled_dataset[k], self.scaled_dataset[f"labels_{k}"] = self.data[k][idx], self.labels[k][idx]

        if preprocess:
            self.scaled_dataset = self.preprocess(self.scaled_dataset, preprocess)

    def create_labels(self):
        labels = dict()

        for i_key, key in enumerate(self.data.keys()):

            anomaly_dataset_i = self.data[key][:]
            print(f"making datasets for {key} anomaly with shape {anomaly_dataset_i.shape}")

            # Predicts anomaly_dataset_i using encoder and defines anomalous labels as 4.0
            labels[key] = np.empty((anomaly_dataset_i.shape[0],1))
            labels[key].fill(4+i_key)

        return labels

    def preprocess(self, data, scaling_filename):
        # Normalizes train and testing features by x' = (x - μ) / σ, where μ, σ are predetermined constants
        tf = True if args.for_transformer else False
        for k in data.keys():
            if not 'label' in k: data[k] = zscore_preprocess(data[k], scaling_file=scaling_filename, for_transformer=tf)

        return data

    def save(self, filename):
        np.savez(filename, **self.scaled_dataset)
        print(f'{filename} successfully saved')

    def report_specs(self):
        '''
        Reports file specs: keys, shape pairs. If divisions, also reports number of samples from each label represented
        in dataset
        '''
        print('File Specs:')
        print(self.scaled_dataset.keys())

        for k in self.scaled_dataset:
            if 'label' in k:
                labels = self.scaled_dataset[k].copy()
                print('labels:', labels)
                labels = labels.reshape((labels.shape[0],))
                label_counts = labels.astype(int)
                label_counts = np.bincount(label_counts)
                label = len(label_counts) - 1
                count = label_counts[-1]
                print(f"Label {label, NAME_MAPPINGS[label]}: {count} occurances")


if __name__=='__main__':

    # Parses terminal command
    parser = ArgumentParser()

    # If not using full data, name of smaller dataset to pull from
    parser.add_argument('background_dataset', type=str)
    parser.add_argument('background_ids', type=str)
    parser.add_argument('anomaly_dataset', type=str)

    parser.add_argument('--scaling-filename', type=str, default=None)
    parser.add_argument('--output-filename', type=str, default=None)
    parser.add_argument('--output-anomaly-filename', type=str, default=None)
    parser.add_argument('--sample-size', type=int, default=-1)
    parser.add_argument('--mix-in-anomalies', action='store_true')

    parser.add_argument('--for-transformer', type=str, default=None)

    args = parser.parse_args()

    # start with making the background dataset
    background_dataset = CLBackgroundDataset(
        args.background_dataset, args.background_ids,
        preprocess=args.scaling_filename,
        divisions=[0.30, 0.30, 0.20, 0.20],
    )
    background_dataset.report_specs()
    if args.for_transformer:
        background_dataset.save(f'{args.output_filename}_tf')
    else:
        background_dataset.save(args.output_filename)

    print()

    # prepare signal datasets
    signal_dataset = CLSignalDataset(
        args.anomaly_dataset,
        preprocess=args.scaling_filename
    )
    signal_dataset.report_specs()
    if args.for_transformer:
        signal_dataset.save(f'{args.output_anomaly_filename}_tf')
    else:
        signal_dataset.save(args.output_anomaly_filename)

