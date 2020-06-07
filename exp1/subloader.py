import torch
import numpy as np
import torchvision.datasets as datasets

class CIFAR10_SubLoader(datasets.CIFAR10):
    def __init__(self, *args, exclude_list=[], **kwargs):
        super(CIFAR10_SubLoader, self).__init__(*args, **kwargs)

        if exclude_list == []:
            return
        remap = {}
        counter = 0
        for i in range(10):
            if i not in exclude_list:
                 remap[i] = counter
                 counter+=1

        labels = np.array(self.targets)
        exclude = np.array(exclude_list).reshape(1, -1)
        mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

        self.data = self.data[mask]
        self.targets = labels[mask].tolist()
        self.targets = [remap[each] for each in self.targets]

class STL10_SubLoader(datasets.STL10):
    def __init__(self, *args, exclude_list=[], **kwargs):
        super(STL10_SubLoader, self).__init__(*args, **kwargs)

        if exclude_list == []:
            return
        remap = {}
        counter = 0
        for i in range(10):
            if i not in exclude_list:
                 remap[i] = counter
                 counter+=1

        labels = np.array(self.labels)
        exclude = np.array(exclude_list).reshape(1, -1)
        mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

        self.data = self.data[mask]
        self.labels = labels[mask].tolist()
        self.labels = [remap[each] for each in self.labels]
