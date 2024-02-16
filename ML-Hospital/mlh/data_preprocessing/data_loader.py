

import torchvision.transforms as transforms
import torch
import numpy as np
from io import RawIOBase
from mlh.data_preprocessing.dataset_preprocessing import prepare_dataset, cut_dataset, prepare_inference_dataset
from torchvision import datasets
from PIL import Image
from tqdm import tqdm
from mlh.data_preprocessing import configs

torch.manual_seed(0)


class GetDataLoader(object):
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.input_shape = args.input_shape

    def parse_dataset(self, dataset, train_transform, test_transform):

        if dataset in configs.SUPPORTED_IMAGE_DATASETS:
            _loader = getattr(datasets, dataset)
            if dataset != "EMNIST":
                train_dataset = _loader(root=self.data_path,
                                        train=True,
                                        transform=train_transform,
                                        download=True)
                test_dataset = _loader(root=self.data_path,
                                       train=False,
                                       transform=test_transform,
                                       download=True)
            else:
                train_dataset = _loader(root=self.data_path,
                                        train=True,
                                        split="byclass",
                                        transform=train_transform,
                                        download=True)
                test_dataset = _loader(root=self.data_path,
                                       train=False,
                                       split="byclass",
                                       transform=test_transform,
                                       download=True)
            dataset = train_dataset + test_dataset

        else:
            raise ValueError("Dataset Not Supported: ", dataset)
        return dataset

    def get_data_transform(self, dataset, use_transform="simple"):
        transform_list = [transforms.Resize(
            (self.input_shape[0], self.input_shape[0])), ]

        if use_transform == "simple":
            transform_list += [transforms.RandomCrop(
                32, padding=4), transforms.RandomHorizontalFlip(), ]

            print("add simple data augmentation!")

        transform_list.append(transforms.ToTensor())

        if dataset in ["MNIST", "FashionMNIST", "EMNIST"]:
            transform_list = [
                transforms.Grayscale(3), ] + transform_list

        transform_ = transforms.Compose(transform_list)
        return transform_


    def get_dataset(self, train_transform, test_transform):
        dataset = self.parse_dataset(
            self.args.dataset, train_transform, test_transform)
        return dataset

    def get_inference_dataset(self, train_transform, test_transform):
        dataset = self.parse_dataset(
            self.args.inference_dataset, train_transform, test_transform)
        return dataset

    def get_data_supervised(self, batch_size=128, num_workers=2):

        train_transform = self.get_data_transform(self.args.dataset)
        test_transform = self.get_data_transform(self.args.dataset)

        dataset = self.get_dataset(train_transform, test_transform)

        target_train, target_inference, target_test, shadow_train, shadow_inference, shadow_test = prepare_dataset(
            dataset, select_num=None)

        print("Preparing dataloader!")
        print("dataset: ", len(dataset))
        print("target_train: %d \t target_inference: %s \t target_test: %s" %
              (len(target_train), len(target_inference), len(target_test)))

        target_train_loader = torch.utils.data.DataLoader(
            target_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        target_inference_loader = torch.utils.data.DataLoader(
            target_inference, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        target_test_loader = torch.utils.data.DataLoader(
            target_test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        shadow_inference_loader = torch.utils.data.DataLoader(
            shadow_inference, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        shadow_test_loader = torch.utils.data.DataLoader(
            shadow_test, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        return target_train_loader, target_inference_loader, target_test_loader, shadow_train_loader, shadow_inference_loader, shadow_test_loader

    def get_ordered_dataset(self, target_dataset):
        """
        Inspired by https://stackoverflow.com/questions/66695251/define-manually-sorted-mnist-dataset-with-batch-size-1-in-pytorch
        """
        label = np.array([row[1] for row in target_dataset])
        sorted_index = np.argsort(label)
        sorted_dataset = torch.utils.data.Subset(target_dataset, sorted_index)
        return sorted_dataset

    def get_label_index(self, target_dataset):
        """
        return starting index for different labels in the sorted dataset
        """
        label_index = []
        start_label = 0
        label = np.array([row[1] for row in target_dataset])
        for i in range(len(label)):
            if label[i] == start_label:
                label_index.append(i)
                start_label += 1
        return label_index

    def get_sorted_data_mixup_mmd(self):

        train_transform = self.get_data_transform(self.args.dataset)
        test_transform = self.get_data_transform(self.args.dataset)
        dataset = self.get_dataset(train_transform, test_transform)

        target_train, target_inference, target_test, shadow_train, shadow_inference, shadow_test = prepare_dataset(
            dataset, select_num=None)

        target_train_sorted = self.get_ordered_dataset(target_train)
        target_inference_sorted = self.get_ordered_dataset(target_inference)
        shadow_train_sorted = self.get_ordered_dataset(shadow_train)
        shadow_inference_sorted = self.get_ordered_dataset(shadow_inference)

        start_index_target_inference = self.get_label_index(
            target_inference_sorted)
        start_index_shadow_inference = self.get_label_index(
            shadow_inference_sorted)

        # note that we set the inference loader's batch size to 1
        target_train_sorted_loader = torch.utils.data.DataLoader(
            target_train_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        target_inference_sorted_loader = torch.utils.data.DataLoader(
            target_inference_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        shadow_train_sorted_loader = torch.utils.data.DataLoader(
            shadow_train_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        shadow_inference_sorted_loader = torch.utils.data.DataLoader(
            shadow_inference_sorted, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)

        return target_train_sorted_loader, target_inference_sorted_loader, shadow_train_sorted_loader, shadow_inference_sorted_loader, start_index_target_inference, start_index_shadow_inference, target_inference_sorted, shadow_inference_sorted

