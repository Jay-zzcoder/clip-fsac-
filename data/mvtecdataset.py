import os
import PIL
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from .anomaly_image_generate import anomaly_generate
from .anomaly_generate_perlin import generate_anomaly_perlin
import matplotlib.pyplot as plt
import numpy as np
from .self_sup_tasks import patch_ex
import cv2
from einops import repeat, rearrange
MVTEC = ['cable','screw', 'capsule','bottle','carpet',   'hazelnut', 'leather',   'grid', 'pill',
                    'transistor', 'metal_nut', 'toothbrush', 'zipper', 'tile', 'wood']
MPDD = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
VisA = ['capsules','candle',  'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2',
             'pcb3', 'pcb4', 'pipe_fryum']
"""
CLSNAMES_map_index = {}
for k, index in zip(CLSNAMES, range(len(CLSNAMES))):
	CLSNAMES_map_index[k] = index
"""

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]




visa_args={'width_bounds_pct': ((0.03, 0.4), (0.03, 0.4)),
           'intensity_logistic_params': (1/12, 24),
           'num_patches': 2,
           'min_object_pct': 0,
           'min_overlap_pct': 0.25,
           'gamma_params':(2, 0.05, 0.03), 'resize':True, 
           'shift':True, 'same':True, 
           'mode':cv2.NORMAL_CLONE, 
           'label_mode':'logistic-intensity',
           'skip_background': None,
           'resize_bounds': (.5, 2)
}




class MVTDataset(Dataset):
    def __init__(
            self,
            source: str,
            classname: str = None,
            resize: int = 240,
            imagesize: int = 240,
            split: str = "train",
            train_val_split: float = 1.0,
            rotate_degrees: int = 0,
            translate: float = 0,
            brightness_factor: float = 0,
            contrast_factor: float = 0,
            saturation_factor: float = 0,
            gray_p: float = 0,
            h_flip_p: float = 0,
            v_flip_p: float = 0,
            scale: float = 0,
            transform=None,
            args=None,
            **kwargs,
    ):
        """
        PyTorch Dataset for Mvtec.

        Args:
            source (str): Path to the data folder.
            classname (str, optional): Name of MVTec class to use. Defaults to None.
            resize (int, optional): Image size for resizing. Defaults to 224.
            imagesize (int, optional): Square size for the loaded image. Defaults to 224.
            split (str, optional): "train" or "test". Defaults to "train".
            train_val_split (float, optional): Split ratio for train and validation. Defaults to 1.0.
            rotate_degrees (int, optional): Degrees for image rotation. Defaults to 0.
            translate (float, optional): Translation factor. Defaults to 0.
            brightness_factor (float, optional): Brightness factor. Defaults to 0.
            contrast_factor (float, optional): Contrast factor. Defaults to 0.
            saturation_factor (float, optional): Saturation factor. Defaults to 0.
            gray_p (float, optional): Grayscale factor. Defaults to 0.
            h_flip_p (float, optional): Horizontal flip probability. Defaults to 0.
            v_flip_p (float, optional): Vertical flip probability. Defaults to 0.
            scale (float, optional): Scaling factor. Defaults to 0.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.args = args
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split
        self.maskpath = os.path.join(self.source, classname, "ground_truth")
        self.exist_mask = os.path.exists(self.maskpath) and len(os.listdir(self.maskpath)) != 0
        self.imgpaths_per_class, self.data_to_iterate = self._get_image_data()
        self.mask_transform = transforms.Compose(
            [transforms.Resize((240, 240), interpolation=transforms.InterpolationMode.BICUBIC),]
             )
        """
        
        self.transform_img = self._create_image_transform(resize, rotate_degrees,
                                                          brightness_factor, contrast_factor, saturation_factor,
                                                          gray_p, h_flip_p, v_flip_p, translate, scale, imagesize)
       
        """
        self.transform_mask = self._create_mask_transform(resize, imagesize)
        self.transform = transform
        self.transform_noise = transforms.ToTensor()
        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        #print(image_path)
        image = PIL.Image.open(image_path).convert("RGB")
        if self.split == "train":
            if self.args.dataset == "visa":  # visa is correct
                img =  np.asarray(image)
                noise_image, mask = patch_ex(img, img, **visa_args)
                #print("mask", mask.shape)
                mask = rearrange(mask, "H W C -> C H W")
                mask_tensor = torch.from_numpy(mask)
                mask_t = self.mask_transform(mask_tensor)
                mask = torch.zeros_like(mask_t)
                #print("mask", mask.shape)
                #print("mask_t", mask_t.shape)
                #noise_image = anomaly_generate(image, classname)[1]
            else:
                _,noise_image,mask,_ = anomaly_generate(image, classname)
                #noise_image = self.transform(noise_image)
                mask_tensor = torch.from_numpy(mask).unsqueeze(0)
                mask_t = self.mask_transform(mask_tensor)
                mask = torch.zeros_like(mask_t)
                #mask = anomaly_generate(image, classname)[2]
            
            plt.figure("str(label)")
            plt.subplot(1, 3, 1).axis('off')
            plt.imshow(image)
            plt.subplot(1, 3, 2).axis('off')
            plt.imshow(noise_image)
            plt.subplot(1, 3, 3).axis('off')
            plt.imshow(mask.squeeze(0))
            save_path = "./result/"+ self.args.dataset + "//" + classname + "/noisy_image/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = "./result/"+ self.args.dataset + "//" + classname + "/noisy_image/"
            plt.close()
            """
            plt.figure("str(label)")
            
            plt.axis('off')
            plt.imshow(image)
            plt.savefig(save_path + str(a) + "_0.svg",bbox_inches='tight', pad_inches = -0.1)
            plt.close()
            plt.figure("str(label)")
            
            plt.axis('off')
            plt.imshow(noise_image)
            plt.savefig(save_path + str(a) + "_1.svg",bbox_inches='tight', pad_inches = -0.1)
            plt.close()
            """
            
            
            
            image = self.transform(image)
            #mask = self._load_and_transform_mask(mask_path)
            noise_image = self.transform(noise_image)
            
        else:
             mask_t = 0
             noise_image=0
             mask = self._load_and_transform_mask(mask_path)
             image = self.transform(image)
        #mask = anomaly_generate(image, classname)[0]
        #noise_image, mask = generate_anomaly_perlin(np.array(image), classname)
        """
        if self.split == "train":
           
            plt.figure("str(label)")
            plt.subplot(1, 2, 1).axis('off')
            plt.imshow(image)
            plt.subplot(1, 2, 2).axis('off')
            plt.imshow(noise_image)
            save_path = "./result/" + classname + "/noisy_image/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            a = np.random.randint(1, 1000)
            plt.savefig(save_path + str(a) + ".jpg")
            plt.close()
        """
            
        
       
        return {
            "image": image,
            "noise_image": noise_image,
            "mask": mask,
            "noisy_mask": mask_t,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "not_anomaly": int(anomaly == "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
            #"cls_id":CLSNAMES_map_index[classname]
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def _get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split)

            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == "train":
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                                                     classname
                                                                 ][anomaly][:train_val_split_idx]
                    elif self.split == "val":
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                                                     classname
                                                                 ][anomaly][train_val_split_idx:]

                if self.split == "test" and anomaly != "good":
                    if self.exist_mask:
                        anomaly_mask_path = os.path.join(self.maskpath, anomaly)
                        anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                        maskpaths_per_class[classname][anomaly] = [
                            os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                        ]
                    else:
                        maskpaths_per_class[classname][anomaly] = None

                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == "test" and anomaly != "good" and self.exist_mask:
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate

    def _create_image_transform(self, resize, rotate_degrees, brightness_factor, contrast_factor,
                                saturation_factor, gray_p, h_flip_p, v_flip_p, translate, scale, imagesize):
        transform_img = [
            transforms.Resize(resize),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees, translate=(translate, translate),
                                    scale=(1.0 - scale, 1.0 + scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        return transforms.Compose(transform_img)

    def _create_mask_transform(self, resize, imagesize):
        transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        return transforms.Compose(transform_mask)

    def _load_and_transform_image(self, image_path):
        image = PIL.Image.open(image_path).convert("RGB")
        return self.transform_img(image)

    def _load_and_transform_mask(self, mask_path):
        if self.split == "test" and mask_path is not None and self.exist_mask:
            mask = PIL.Image.open(mask_path)
            return self.transform_mask(mask)
        elif self.exist_mask:
            return torch.zeros([1, *self.imagesize[1:]])
        else:
            return []

# Usage example:
# dataset = MVTecDataset(source="path_to_data_folder", classname="some_class")
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
def create_mvtect_dataloader(classname, source="", split="train", batch_size=8, num_workers=1, transform=None, args=None):
    """
    Create a DataLoader for MVTec dataset.

    Args:
        classname (str): Name of the MVTec class to use.
        source (str, optional): Path to the data folder. Defaults to an empty string.
        split (str, optional): "train" or "test". Defaults to "train".
        batch_size (int, optional): Batch size. Defaults to 8.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 4.

    Returns:
        DataLoader: DataLoader for the MVTec dataset.
    """
    dataloader = DataLoader(
        dataset=MVTDataset(source=source, classname=classname, split=split,transform=transform, args=args),
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
        
    )

    return dataloader