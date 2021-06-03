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

class CIFAR100_SubLoader(datasets.CIFAR100):
    def __init__(self, *args, superclass = "aquatic_mammals", **kwargs):
        super(CIFAR100_SubLoader, self).__init__(*args, **kwargs)
        class_dict = {"aquatic_mammals":["beaver", "dolphin", "otter", "seal", "whale"],
                      "fish":["aquarium_fish", "flatfish", "ray", "shark", "trout"],
                      "flowers":["orchid", "poppy", "rose", "sunflower", "tulip"],
                      "food_containers":["bottle", "bowl", "can", "cup", "plate"],
                      "fruit_and_vegetables":["apple", "mushroom", "orange", "pear", "sweet_pepper"],
                      "household_electrical_devices": ["clock", "keyboard", "lamp", "telephone", "television"],
                      "household_furniture":["bed", "chair", "couch", "table", "wardrobe"],
                      "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
                      "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
                      "large_man-made_outdoor_things":["bridge", "castle", "house", "road", "skyscraper"],
                      "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
                      "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
                      "medium_sized_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
                      "non_insect_invertebrates":["crab", "lobster", "snail", "spider", "worm"],
                      "people":["baby", "boy", "girl", "man", "woman"],
                      "reptiles":["crocodile", "dinosaur", "lizard", "snake", "turtle"],
                      "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
                      "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
                      "vehicles_1":["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
                      "vehicles_2":["lawn_mower", "rocket", "streetcar", "tank", "tractor"]
        }
        
        include_list = [self.classes.index(each) for each in class_dict[superclass]]
        remap = {}
        counter = 0
        for i in range(100):
            if i in include_list:
                 remap[i] = counter
                 counter+=1
        
        labels = np.array(self.targets)
        include = np.array(include_list).reshape(1, -1)
        mask = (labels.reshape(-1, 1) == include).any(axis=1)

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
