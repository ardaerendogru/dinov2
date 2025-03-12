# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.



from torchvision import transforms
from torchvision.transforms import RandAugment

from .transforms import (
    make_normalize_transform,
)




class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        global_crops_size=224,
    ):
        self.global_crops_scale = global_crops_scale
        self.global_crops_size = global_crops_size


        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        random_erasing = transforms.RandomErasing(
                    p=0.25,  # erase_prob
                    scale=(0.02, 1/3),  # min_area_ratio, max_area_ratio
                    ratio=(0.3, 3.3),   # default aspect ratio range
                    inplace=False
                )


        # Create RandAugment transform
        rand_augment = RandAugment(
            num_ops=9,  # Number of augmentation transformations to apply sequentially
            magnitude=9,  # Magnitude for all the transformations
            num_magnitude_bins=31,  # The number of different magnitude values
            interpolation=transforms.InterpolationMode.BILINEAR,  # Interpolation mode
            fill=None  # Pixel fill value for operations that require it
        )

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([rand_augment, self.normalize, random_erasing])

    def __call__(self, image):

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)


        return global_crop_1
