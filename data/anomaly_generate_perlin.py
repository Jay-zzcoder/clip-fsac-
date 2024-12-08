import cv2
import os
import numpy as np
from glob import glob
import torch
import torchvision.transforms as transforms
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from typing import List, Tuple
from einops import rearrange
import math
from torch.utils.data import Dataset, DataLoader


anomaly_config = {
    "hazelnut": {
        "use_mask"     : True,
        "bg_threshold" : 50,
        "bg_reverse"   : True
    },
    "leather": {
        "use_mask"     : False,
        "bg_threshold" : None,
        "bg_reverse"   : None
    },
    "metal_nut": {
        "use_mask"     : True,
        "bg_threshold" : 40,
        "bg_reverse"   : True
    },
    "tile": {
        "use_mask"     : False,
        "bg_threshold" : None,
        "bg_reverse"   : None
    },
    "wood": {
        "use_mask"     : False,
        "bg_threshold" : None,
        "bg_reverse"   : None
    },
    "grid": {
        "use_mask"     : False,
        "bg_threshold" : None,
        "bg_reverse"   : None
    },
    "cable": {
        "use_mask"     : True,
        "bg_threshold" : 150,
        "bg_reverse"   : True
    },
    "capsule": {
        "use_mask"     : True,
        "bg_threshold" : 120,
        "bg_reverse"   : False
    },
    "transistor": {
        "use_mask"     : True,
        "bg_threshold" : 90,
        "bg_reverse"   : False
    },
    "carpet": {
        "use_mask"     : False,
        "bg_threshold" : None,
        "bg_reverse"   : None
    },
    "bottle": {
        "use_mask"     : True,
        "bg_threshold" : 250,
        "bg_reverse"   : False
    },
    "screw": {
        "use_mask"     : True,
        "bg_threshold" : 110,
        "bg_reverse"   : False
    },
    "zipper": {
        "use_mask"     : True,
        "bg_threshold" : 100,
        "bg_reverse"   : False
    },
    "pill": {
        "use_mask"     : True,
        "bg_threshold" : 100,
        "bg_reverse"   : True
    },
    "toothbrush": {
        "use_mask"     : True,
        "bg_threshold" : 30,
        "bg_reverse"   : True
    }
}


def lerp_np(x,y,w):
    fin_out = (y-x)*w + x
    return fin_out

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients,d[0],axis=0),d[1],axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim=-1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0],
                                                                                                              0).repeat_interleave(
        d[1], 1)
    dot = lambda grad, shift: (
                torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])

    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise


def generate_anomaly_perlin(img: np.ndarray,
                     classname: str,
                     texture_img_list: list = None,
                     resize=(1024, 1024),
                     transparency_range: List[float] = [0.15, 1.],
                     perlin_scale=6,
                     min_perlin_scale=1,
                     perlin_noise_threshold=0.9,
                     structure_grid_size: str = 8,
                     ) -> List[np.ndarray]:

    # step 1. generate mask
    use_mask = anomaly_config[classname]["use_mask"]
    bg_threshold = anomaly_config[classname]["bg_threshold"]
    bg_reverse = anomaly_config[classname]["bg_reverse"]
    ## target foreground mask
    if use_mask:
        target_foreground_mask = generate_target_foreground_mask(img=img,
                                                                 bg_threshold=bg_threshold,
                                                                 bg_reverse=bg_reverse)
    else:
        target_foreground_mask = np.ones(resize)

    ## perlin noise mask
    perlin_noise_mask = generate_perlin_noise_mask(resize=resize,
                                                   min_perlin_scale=min_perlin_scale,
                                                   perlin_scale=perlin_scale,
                                                   perlin_noise_threshold=perlin_noise_threshold)

    ## mask
    mask = perlin_noise_mask * target_foreground_mask
    mask_expanded = np.expand_dims(mask, axis=2)

    # step 2. generate texture or structure anomaly

    ## anomaly source
    anomaly_source_img = anomaly_source(img=img,
                                        resize=resize,
                                        texture_img_list=texture_img_list,
                                        structure_grid_size=structure_grid_size)

    ## mask anomaly parts
    factor = np.random.uniform(*transparency_range, size=1)[0]
    anomaly_source_img = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)

    # step 3. blending image and anomaly source
    anomaly_source_img = ((- mask_expanded + 1) * img) + anomaly_source_img

    return (anomaly_source_img.astype(np.uint8), mask)


def generate_target_foreground_mask(img: np.ndarray, bg_threshold, bg_reverse) -> np.ndarray:
    # convert RGB into GRAY scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # generate binary mask of gray scale image
    _, target_background_mask = cv2.threshold(img_gray, bg_threshold, 255, cv2.THRESH_BINARY)
    target_background_mask = target_background_mask.astype(np.bool).astype(np.int)

    # invert mask for foreground mask
    if bg_reverse:
        target_foreground_mask = target_background_mask
    else:
        target_foreground_mask = -(target_background_mask - 1)

    return target_foreground_mask


def generate_perlin_noise_mask(resize, min_perlin_scale, perlin_scale, perlin_noise_threshold) -> np.ndarray:
    # define perlin noise scale
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

    # generate perlin noise
    perlin_noise = rand_perlin_2d_np((resize[0], resize[1]), (perlin_scalex, perlin_scaley))

    # apply affine transform
    rot = iaa.Affine(rotate=(-90, 90))
    perlin_noise = rot(image=perlin_noise)

    # make a mask by applying threshold
    mask_noise = np.where(
        perlin_noise > perlin_noise_threshold,
        np.ones_like(perlin_noise),
        np.zeros_like(perlin_noise)
    )

    return mask_noise


def anomaly_source(img: np.ndarray, resize, structure_grid_size, texture_img_list: list = None,) -> np.ndarray:
    p = np.random.uniform() if texture_img_list else 1.0
    if p < 0.5:
        idx = np.random.choice(len(texture_img_list))
        anomaly_source_img = _texture_source(texture_img_path=texture_img_list[idx], resize=resize)
    else:
        anomaly_source_img = _structure_source(img=img, resize=resize, structure_grid_size=structure_grid_size)

    return anomaly_source_img


def _texture_source(texture_img_path: str, resize) -> np.ndarray:
    texture_source_img = cv2.imread(texture_img_path)
    texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
    texture_source_img = cv2.resize(texture_source_img, dsize=(resize[1], resize[0])).astype(np.float32)

    return texture_source_img


def _structure_source(img: np.ndarray, resize, structure_grid_size) -> np.ndarray:
    structure_source_img = rand_augment()(image=img)

    assert resize[0] % structure_grid_size == 0, 'structure should be devided by grid size accurately'
    grid_w = resize[1] // structure_grid_size
    grid_h = resize[0] // structure_grid_size

    structure_source_img = rearrange(
        tensor=structure_source_img,
        pattern='(h gh) (w gw) c -> (h w) gw gh c',
        gw=grid_w,
        gh=grid_h
    )
    disordered_idx = np.arange(structure_source_img.shape[0])
    np.random.shuffle(disordered_idx)

    structure_source_img = rearrange(
        tensor=structure_source_img[disordered_idx],
        pattern='(h w) gw gh c -> (h gh) (w gw) c',
        h=structure_grid_size,
        w=structure_grid_size
    ).astype(np.float32)

    return structure_source_img

def rand_augment():
    augmenters = [
        iaa.GammaContrast((0.5, 2.0), per_channel=True),
        iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
        iaa.pillike.EnhanceSharpness(),
        iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
        iaa.Solarize(0.5, threshold=(32, 128)),
        iaa.Posterize(),
        iaa.Invert(),
        iaa.pillike.Autocontrast(),
        iaa.pillike.Equalize(),
        iaa.Affine(rotate=(-45, 45))
    ]
    aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
    aug = iaa.Sequential([
        augmenters[aug_idx[0]],
        augmenters[aug_idx[1]],
        augmenters[aug_idx[2]]
    ])
    return aug

