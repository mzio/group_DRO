"""
Colored MNIST Dataset

Loadable with:
```
loaders = load_colored_mnist(args, 
                             train_shuffle=True,
                             transform=None)
train_loader, val_loader, test_loader = loaders                 
```

See colored_mnist.md for more details on loading data arguments.

`train_val_split` function imported is reproduced here:
```
def train_val_split(dataset, val_split, seed):
    '''
    Compute indices for train and val splits
    
    Args:
    - dataset (torch.utils.data.Dataset): Pytorch dataset
    - val_split (float): Fraction of dataset allocated to validation split
    - seed (int): Reproducibility seed
    Returns:
    - train_indices, val_indices (np.array, np.array): Dataset indices
    '''
    train_ix = int(np.round(val_split * len(dataset)))
    all_indices = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    train_indices = all_indices[train_ix:]
    val_indices = all_indices[:train_ix]
    return train_indices, val_indices
```
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from datasets import train_val_split
# from utils.visualize import plot_data_batch


class ColoredMNIST(Dataset):
    """
    Colored MNIST dataset - labels spuriously correlated with color
    - We store the label, the spurious attribute, and subclass labels if applicable

    Args:
    - data (torch.Tensor): MNIST images
    - targets (torch.Tensor): MNIST original labels
    - train_classes (list[]): List of lists describing how to organize labels
                                - Each inner list denotes a group, i.e. 
                                they all have the same classification label
                                - Any labels left out are excluded from training set
    - train (bool): Training or test dataset
    - p_correlation (float): Strength of spurious correlation, in [0, 1]
    - test_shift (str): How to organize test set, from 'random', 'same', 'new'
    - cmap (str): Colormap for coloring MNIST digits
    - flipped (bool): If true, color background and keep digit black
    - transform (torchvision.transforms): Image transformations
    - args (argparse): Experiment arguments
    Returns:
    - __getitem__() returns tuple of image, label, and the index, which can be used for
                    looking up additional info (e.g. subclass label, spurious attribute)
    """

    def __init__(self, data, targets, train_classes=[[1], [2], [3], [4], [5], [6], [7], [8], [9]],
                 train=True, p_correlation=0.8, test_shift='random', cmap='hsv',
                 flipped=False, transform=None, args=None):
        self.args = args
        # Initialize classes
        self.class_map = self._init_class_map(train_classes)
        self.classes = list(self.class_map.keys())
        self.new_classes = np.unique(list(self.class_map.values()))
        
        self.target_name = 'digit'
        self.n_confounders = 1  # 1  # 5  # 1
        self.confounder_names = ['color']
        
        # Groups
        self.n_groups = len(train_classes) ** 2
        self.n_classes = len(train_classes)
        
        print(f'n_groups: {self.n_groups}')
        print(f'self.n_classes: {self.n_classes}')

        self.test_classes = [x for x in np.unique(
            targets) if x not in self.classes]
        self.p_correlation = p_correlation
        # Setup spurious correlation ratios per class
        if args.p_corr_by_class is not None:
            self.p_correlation = args.p_corr_by_class
        else:
            self.p_correlation = [p_correlation] * len(self.new_classes)
        self.train = train
        self.test_shift = test_shift
        self.transform = transform

        # Filter for train_classes
        class_filter = torch.stack([(targets == i)
                                    for i in self.classes]).sum(dim=0)
        self.targets = targets[class_filter > 0]
        data = data[class_filter > 0]

        self.targets_all = {'spurious': np.zeros(len(self.targets), dtype=int),
                            'sub_target': copy.deepcopy(self.targets)}
        # Update targets
        self.targets = torch.tensor([self.class_map[t.item()] for t in self.targets],
                                    dtype=self.targets.dtype)
        self.targets_all['target'] = self.targets.numpy()
        # Group_idx is whether the datapoint is spuriously correlated or not
        self.targets_all['group_idx'] = (
            self.targets_all['target'] == self.targets_all['spurious']).astype(int)
        # For now just consider if spurious attribute is correlated with ground-truth
        self.group_labels = ['no spurious correlation',
                             'spurious correlation']
        # Colors + Data
        self.colors = self._init_colors(cmap)
        if flipped:
            data = 255 - data
        if data.shape[1] != 3:   # Add RGB channels
            data = data.unsqueeze(1).repeat(1, 3, 1, 1)
        self.data = self._init_data(data)
        self.spurious_group_names = self.colors
        # Adjust in case data was resampled for class imbalance
        if self.args.train_class_ratios is not None and self.train is True:
            self.targets = self.targets[self.selected_indices]
            for k in self.targets_all:
                self.targets_all[k] = self.targets_all[k][self.selected_indices]
                
        # Make color the ground-truth, digit the spurious attribute
        if args.replicate in np.arange(100, 200) and self.train is False:
            self.targets = torch.tensor(self.targets_all['spurious'],
                                        dtype=self.targets.dtype)
            
#         group_array = []
#         y_array = []

#         for x,y,g in self:
#             group_array.append(g)
#             y_array.append(y)
            
        # target_spurious_to_group_ix = np.arange(25).reshape((5, 5)).astype('int')
        
        target_spurious_to_group_ix = np.arange(self.n_groups).reshape((self.n_classes, self.n_classes)).astype('int')
        
        group_array = []
        for ix in range(len(self.targets_all['target'])):
            y = self.targets_all['target'][ix]
            a = self.targets_all['spurious'][ix]
            group_array.append(target_spurious_to_group_ix[y][a])
        group_array = np.array(group_array)
            
        # group_array = (self.targets_all['target']*(self.n_groups/2) + self.targets_all['spurious']).astype('int')
        print(np.unique(group_array, return_counts=True))
        
        self.group_array = group_array
        self.y_array = self.targets_all['target']
        
        self._group_array = torch.LongTensor(group_array)
        self._y_array = torch.LongTensor(self.targets_all['target'])
        self._group_counts = (torch.arange(self.n_groups).unsqueeze(1)==self._group_array).sum(1).float()
        self._y_counts = (torch.arange(self.n_classes).unsqueeze(1)==self._y_array).sum(1).float()
        

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        g = self.group_array[idx]
        if self.transform:
            x = self.transform(x)
        return (x, y, g)

    def _init_class_map(self, classes):
        class_map = {}
        for c_ix, targets in enumerate(classes):
            for t in targets:
                class_map[t] = c_ix
        return class_map

    def _init_colors(self, cmap):
        # Initialize list of RGB color values
        try:
            cmap = cm.get_cmap(cmap)
        except ValueError:  # single color
            cmap = self._get_single_color_cmap(cmap)
        cmap_vals = np.arange(0, 1, step=1 / len(self.new_classes))
        colors = []
        for ix, c in enumerate(self.new_classes):
            rgb = cmap(cmap_vals[ix])[:3]
            rgb = [int(np.float(x)) for x in np.array(rgb) * 255]
            colors.append(rgb)
        return colors

    def _get_single_color_cmap(self, c):
        rgb = to_rgb(c)
        r1, g1, b1 = rgb
        cdict = {'red':   ((0, r1, r1),
                           (1, r1, r1)),
                 'green': ((0, g1, g1),
                           (1, g1, g1)),
                 'blue':  ((0, b1, b1),
                           (1, b1, b1))}
        cmap = LinearSegmentedColormap('custom_cmap', cdict)
        return cmap

    def _init_data(self, data):
        np.random.seed(self.args.seed)
        self.selected_indices = []
        pbar = tqdm(total=len(self.targets), desc='Initializing data')
        for ix, c in enumerate(self.new_classes):
            class_ix = np.where(self.targets == c)[0]
            # Introduce class imbalance
            if self.args.train_class_ratios is not None and self.train is True:
                class_size = int(np.round(
                    len(class_ix) * self.args.train_class_ratios[ix][0]))
                class_ix = np.random.choice(
                    class_ix, size=class_size, replace=False)
                self.selected_indices.append(class_ix)
            is_spurious = np.random.binomial(1, self.p_correlation[ix],
                                             size=len(class_ix))
            for cix_, cix in enumerate(class_ix):
                # Replace pixels
                pixels_r = np.where(
                    np.logical_and(data[cix, 0, :, :] >= 120,
                                   data[cix, 0, :, :] <= 255))
                # May refactor this out as a separate function later
                if self.train or self.test_shift == 'iid':
                    color_ix = (ix if is_spurious[cix_] else
                                np.random.choice([
                                    x for x in np.arange(len(self.colors)) if x != ix]))
                elif 'shift' in self.test_shift:
                    n = int(self.test_shift.split('_')[-1])
                    color_ix = (ix + n) % len(self.new_classes)
                else:
                    color_ix = np.random.randint(len(self.colors))
                color = self.colors[color_ix]
                data[cix, :, pixels_r[0], pixels_r[1]] = (
                    torch.tensor(color, dtype=torch.uint8).unsqueeze(1).repeat(1, len(pixels_r[0])))
                self.targets_all['spurious'][cix] = int(color_ix)
                pbar.update(1)
        if self.args.train_class_ratios is not None and self.train is True:
            self.selected_indices = np.concatenate(self.selected_indices)
            return data[self.selected_indices].float() / 255
        return data.float() / 255  # For normalization

    def get_dataloader(self, batch_size, shuffle, num_workers):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    def group_str(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes)
        c = group_idx % (self.n_groups//self.n_classes)

        group_name = f'{self.target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        print(bin_str)
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name
    
    def group_counts(self):
        return self._group_counts

    def class_counts(self):
        return self._y_counts

    def input_size(self):
        for x,y,g in self:
            return x.size()


def load_colored_mnist(args, train_shuffle=True, transform=None, 
                       datasets=False):
    """
    Default dataloader setup for Colored MNIST

    Args:
    - args (argparse): Experiment arguments
    - transform (torchvision.transforms): Image transformations
    Returns:
    - (train_loader, test_loader): Tuple of dataloaders for train and test
    """
    mnist_train = torchvision.datasets.MNIST(
        root=args.data_path, train=True, download=False)
    mnist_test = torchvision.datasets.MNIST(
        root=args.data_path, train=False, download=False)

    transform = (transforms.Compose([transforms.Resize(40),
                                     transforms.RandomCrop(32, padding=0),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                 if transform is None else transform)
    
    # Split original train set into train and val
    train_indices, val_indices = train_val_split(mnist_train, args.val_split, args.seed)
    train_data = mnist_train.data[train_indices]
    train_targets = mnist_train.targets[train_indices]
    val_data = mnist_train.data[val_indices]
    val_targets = mnist_train.targets[val_indices]
    
    colored_mnist_train = ColoredMNIST(data=train_data, targets=train_targets,
                                       train_classes=args.train_classes, train=True,
                                       p_correlation=args.p_correlation,
                                       test_shift=args.test_shift,
                                       cmap=args.data_cmap,
                                       transform=transform,
                                       flipped=args.flipped,
                                       args=args)
    # Val set is setup with same data distribution as test set by convention.
    # But maybe want to make it the same data distribution as train set?
    colored_mnist_val = None
    if len(val_data) > 0:
        colored_mnist_val = ColoredMNIST(data=val_data, targets=val_targets,
                                         train_classes=args.train_classes, train=False,
                                         p_correlation=args.p_correlation,
                                         test_shift=args.test_shift,
                                         cmap=args.data_cmap,
                                         transform=transform,
                                         flipped=args.flipped,
                                         args=args)
        
    test_cmap = args.data_cmap if args.test_cmap == '' else args.test_cmap
    test_p_corr = args.p_correlation if args.test_cmap == '' else 1.0
    colored_mnist_test = ColoredMNIST(data=mnist_test.data, targets=mnist_test.targets,
                                      train_classes=args.train_classes, train=False,
                                      p_correlation=test_p_corr,
                                      test_shift=args.test_shift,
                                      cmap=test_cmap,
                                      transform=transform,
                                      flipped=args.flipped,
                                      args=args)
    
    if datasets:
        return colored_mnist_train, colored_mnist_val, colored_mnist_test
    
    train_loader = DataLoader(colored_mnist_train, batch_size=args.bs_trn,
                              shuffle=train_shuffle, num_workers=args.num_workers)
    val_loader   = DataLoader(colored_mnist_val, batch_size=args.bs_val,
                              shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(colored_mnist_test, batch_size=args.bs_val,
                              shuffle=False, num_workers=args.num_workers)
    # Update args.num_classes
    args.num_classes = len(colored_mnist_train.new_classes)
    return train_loader, val_loader, test_loader


def imshow(img, mean=0.5, std=0.5):
    """
    Visualize data batches
    """
    img = img * std + mean  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def visualize_colored_mnist(dataloader, num_datapoints, title, args, save,
                            save_id, ftype='png', target_type='target'):
    """
    Visualize dataset.
    Args:
    - target_type (str): Which labels to visualize by, e.g. 'group_idx', 'target', 'spurious'
    """
    # Filter for selected datapoints (in case we use SubsetRandomSampler)
    try:
        subset_indices = dataloader.sampler.indices
        targets = dataloader.dataset.targets_all[target_type][subset_indices]
        subset = True
    except AttributeError:
        targets = dataloader.dataset.targets_all[target_type]
        subset = False
    all_data_indices = []
    for class_ in np.unique(targets):
        class_indices = np.where(targets == class_)[0]
        if subset:
            class_indices = subset_indices[class_indices]
        all_data_indices.extend(class_indices[:num_datapoints])
    
#     plot_data_batch([dataloader.dataset.__getitem__(ix)[0] for ix in all_data_indices],
#                     mean=0,
#                     std=1, nrow=8, title=title,
#                     args=args, save=save, save_id=save_id, ftype=ftype)
    
    
# Refactor for modularity
def load_dataloaders(args, train_shuffle=True, transform=None):
    return load_colored_mnist(args, train_shuffle, transform)


def visualize_dataset(dataloader, num_datapoints, title, args, save,
                            save_id, ftype='png', target_type='target'):
    return visualize_colored_mnist(dataloader, num_datapoints, title, 
                                   args, save, save_id, ftype, target_type)

