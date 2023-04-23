import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch_geometric.loader import DataLoader


def get_sample_weights_from_class(dataset):
    class_label = torch.IntTensor([i.y.item() for i in dataset])
    unique_class, bins = torch.unique(class_label, return_counts=True)
    weight_bins = 1/bins
    weights = weight_bins[class_label]
    return weights


def train_val_test_split_list(datasets:list=[], train_ratio=0.8, val_ratio=0.1):
    train_data_list = []
    val_data_list = []
    test_data_list = []
    for dataset in datasets:
        dataset = dataset.shuffle()
        train_data_list.append(dataset[:round(len(dataset)*train_ratio)])
        val_data_list.append(dataset[round(len(dataset)*train_ratio):round(len(dataset)*(train_ratio+val_ratio))])
        test_data_list.append(dataset[round(len(dataset)*(train_ratio+val_ratio)):])
    return train_data_list, val_data_list, test_data_list


def train_data_loaders_list(train_data_list:list=[], datatype_list:list=[], batch_size=4):
    if batch_size==None:
        batch_size=round(64/len(train_data_list))
    train_data_loaders_list = []
    for dataset, task in zip(train_data_list, datatype_list):
        if task == 'r':
            train_data_loaders_list.append(DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True))
        elif task == 'c':
            weights = get_sample_weights_from_class(dataset)
            sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
            train_data_loaders_list.append(DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler))
    return train_data_loaders_list


def val_test_data_loaders_list(val_test_data_list:list=[], batch_size=4):
    if batch_size==None:
        batch_size=round(64/len(val_test_data_list))
    val_test_data_loaders_list = []
    for dataset in val_test_data_list:
        val_test_data_loaders_list.append(DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True))
    return val_test_data_loaders_list


class zip_loaders(object):
    def __init__(self, loaders_list):
        self.loaders_list = loaders_list
        self.loaders_iter_list =[iter(i) for i in loaders_list]

    def next(self):
        batch_list = []
        for idx, loader in enumerate(self.loaders_iter_list):
            try:
                batch = next(loader)
            except StopIteration:
                loader = iter(self.loaders_list[idx])
                self.loaders_iter_list[idx] = loader
                batch = next(loader)
            batch_list.append(batch)
        return batch_list








