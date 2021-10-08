import torch
import torch.utils.data as data
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as torchdata

class DsetThreeChannels(data.Dataset):
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, index):
        image, label = self.dset[index]
        return image.repeat(3, 1, 1), label

    def __len__(self):
        return len(self.dset)

def prepare_dataset(dataset_name, image_size, channels, path):

    #################################################################

    if dataset_name == 'usps':
        from usps import USPS
        tr_dataset = USPS(path+'/usps', 
            download=True, 
            train=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        print('USPS train set size: %d' %(len(tr_dataset)))
        te_dataset = USPS(path+'/usps', 
            download=True, 
            train=False, 
            transform=transforms.Compose([transforms.Resize(image_size)]))
        print('USPS test set size: %d' %(len(te_dataset)))

        if channels == 3:
            tr_dataset = DsetThreeChannels(tr_dataset)
            te_dataset = DsetThreeChannels(te_dataset)

    #################################################################

    elif dataset_name == 'mnist':
        tr_dataset = torchvision.datasets.MNIST(path+'/mnist', 
            download=True, 
            train=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        te_dataset = torchvision.datasets.MNIST(path+'/mnist', 
            download=True, 
            train=False, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        if channels == 3:
            tr_dataset = DsetThreeChannels(tr_dataset)
            te_dataset = DsetThreeChannels(te_dataset)

    #################################################################

    elif dataset_name == 'mnistrot':
        tr_dataset = torchvision.datasets.MNIST(path+'/mnistrot', 
            download=True, 
            train=True, 
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomRotation(30, fill=(0,))]))

        te_dataset = torchvision.datasets.MNIST(path+'/mnistrot', 
            download=True, 
            train=False, 
            transform=transforms.Compose([transforms.Resize(image_size),
                transforms.RandomRotation(30, fill=(0,))]))

        if channels == 3:
            tr_dataset = DsetThreeChannels(tr_dataset)
            te_dataset = DsetThreeChannels(te_dataset)

    #################################################################

    elif dataset_name == 'mnistm':
        from mnistm import MNISTM
        tr_dataset = MNISTM(path+'/mnistm', 
            download=True, 
            train=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        te_dataset = MNISTM(path+'/mnistm', 
            download=True, 
            train=False, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

    #################################################################

    elif dataset_name == 'svhn':
        tr_dataset = torchvision.datasets.SVHN(root=path+'/svhn', 
            split='train', 
            download=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        print('SVHN basic set size: %d' %(len(tr_dataset)))
        te_dataset = torchvision.datasets.SVHN(root=path+'/svhn', 
            split='test', 
            download=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

    #################################################################

    elif dataset_name == 'svhn_extra':
        tr_dataset_basic = torchvision.datasets.SVHN(root=path+'/svhn', 
            split='train', 
            download=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        print('SVHN basic set size: %d' %(len(tr_dataset_basic)))

        tr_dataset_extra = torchvision.datasets.SVHN(root=path+'/svhn', 
            split='extra', 
            download=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        print('SVHN extra set size: %d' %(len(tr_dataset_extra)))

        tr_dataset = torchdata.ConcatDataset((tr_dataset_basic, tr_dataset_extra))

        te_dataset = torchvision.datasets.SVHN(root=path+'/svhn', 
            split='test', 
            download=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

    #################################################################

    elif dataset_name == 'cifar10':
        tr_dataset = torchvision.datasets.CIFAR10(path+'/', 
            train=True, 
            download=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        te_dataset = torchvision.datasets.CIFAR10(path+'/', 
            train=False, 
            download=True, 
            transform=transforms.Compose([transforms.Resize(image_size)]))

        from modify_cifar_stl import modify_cifar
        modify_cifar(tr_dataset)
        modify_cifar(te_dataset)

    #################################################################

    elif dataset_name == 'stl10':
        tr_dataset = torchvision.datasets.STL10(path+'/', split='train', download=True, transform=transforms.Compose([
                                                    transforms.Resize(image_size),
                                                    transforms.ToTensor(),
                                         ]))
        te_dataset = torchvision.datasets.STL10(path+'/', split='test', 
            ownload=True, 
            transform=transforms.Compose([transforms.Resize(image_size),
                transforms.ToTensor()]))

        from modify_cifar_stl import modify_stl
        modify_stl(tr_dataset)
        modify_stl(te_dataset)

    #################################################################

    else:
        raise ValueError('Dataset %s not found!' %(dataset_name))

    #################################################################
    
    return tr_dataset, te_dataset

class create_dataset(data.Dataset):
    def __init__(self, args, is_train):

        self.train = is_train

        sc_tr_dataset, sc_te_dataset = prepare_dataset(args.source, args.image_size, args.channels, path=args.dset_dir)
        tg_tr_dataset, tg_te_dataset = prepare_dataset(args.target, args.image_size, args.channels, path=args.dset_dir)
        
        if self.train:
            self.datalist_src = sc_tr_dataset
            self.datalist_target = tg_tr_dataset
        
        if not self.train:
            self.datalist_src = sc_te_dataset
            self.datalist_target = tg_te_dataset

        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.source_larger = len(self.datalist_src) > len(self.datalist_target)
        self.n_smallerdataset = len(self.datalist_target) if self.source_larger else len(self.datalist_src)


    def __len__(self):
        return np.maximum(len(self.datalist_src), len(self.datalist_target))

    def shuffledata(self):
        self.datalist_src = [self.datalist_src[ij] for ij in torch.randperm(len(self.datalist_src))]
        self.datalist_target = [self.datalist_target[ij] for ij in torch.randperm(len(self.datalist_target))]

    def __getitem__(self, index):
        index_src = index if self.source_larger else index % self.n_smallerdataset
        index_target = index if not self.source_larger else index % self.n_smallerdataset

        image_source, label_source = self.datalist_src[index_src]
        image_source = self.totensor(image_source)
        image_source = self.normalize(image_source)

        image_target, label_target = self.datalist_target[index_target]
        image_target = self.totensor(image_target)
        image_target = self.normalize(image_target)        

        return image_source, label_source, image_target, label_target

def create_dataloader(dataset, args, is_train):
    num_workers = args.num_workers
    batch_size = args.batch_size
    shuffle = True if (is_train == True) else False
    drop_last = True if (is_train == True) else False
    return torch.utils.data.DataLoader(dataset, 
                                       batch_size=batch_size, 
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       drop_last=drop_last,
                                       pin_memory=False
                                       )
