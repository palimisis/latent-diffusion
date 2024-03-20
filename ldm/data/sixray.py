import os
import yaml
import pickle
import shutil
import tarfile
import glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset

import taming.data.utils as tdu
from taming.data.imagenet import str_to_indices, give_synsets_from_indices, download, retrieve
from taming.data.imagenet import ImagePaths

from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light


import pwd
from torch.utils.data import Dataset
import pandas as pd
import torch
from torchvision import transforms


# def synset2idx(path_to_yaml="data/index_synset.yaml"):
#   with open(path_to_yaml) as f:
#     di2s = yaml.load(f)
#   return dict((v, k) for k, v in di2s.items())


# class SixRayBase(Dataset):
#   def __init__(self, config=None, is_train: bool = True):
#     self.is_train = is_train
#     self.config = config or OmegaConf.create()
#     if not type(self.config) == dict:
#       self.config = OmegaConf.to_container(self.config)
#     self.keep_orig_class_label = self.config.get(
#         "keep_orig_class_label", False)
#     # if False we skip loading & processing images and self.data contains filepaths
#     self.process_images = True
#     self._prepare()
#     self._prepare_synset_to_human()
#     # self._prepare_idx_to_synset()
#     self._prepare_human_to_integer_label()
#     self._load()

#   def __len__(self):
#     return len(self.data)

#   def __getitem__(self, i):
#     return self.data[i]

#   def _prepare(self):
#     raise NotImplementedError()

#   def _filter_relpaths(self, relpaths):
#     ignore = set([
#         "n06596364_9591.JPEG",
#     ])
#     relpaths = [
#         rpath for rpath in relpaths if not rpath.split("/")[-1] in ignore]
#     if "sub_indices" in self.config:
#       indices = str_to_indices(self.config["sub_indices"])
#       synsets = give_synsets_from_indices(
#           indices, path_to_yaml=self.idx2syn)  # returns a list of strings
#       self.synset2idx = synset2idx(path_to_yaml=self.idx2syn)
#       files = []
#       for rpath in relpaths:
#         syn = rpath.split("/")[0]
#         if syn in synsets:
#           files.append(rpath)
#       return files
#     else:
#       return relpaths

#   def _prepare_synset_to_human(self):
#     # path = "/home/panagiotisa/datasets/sixray/ImageSet/10/SixRay_train"
#     self.human_dict = os.path.join(self.data_specific, "synset_human.txt")
#     csv_file_path = os.path.join(
#         self.data_specific, "train.csv" if self.is_train else "test.csv")

#     if not os.path.exists(self.human_dict):
#       all_images_df = pd.read_csv(csv_file_path)

#       with open(self.human_dict, 'a') as file:
#         for index, row in all_images_df.iterrows():
#           name = row['name']
#           if name.startswith('N'):
#             file.write(f"{name}\tNo class\n")
#           else:
#             classes = [col for col, val in row.items() if val ==
#                        1 and col != 'name']
#             class_string = ', '.join(classes)
#             file.write(f"{name}\t{class_string}\n")

#   def _prepare_idx_to_synset(self):
#     URL = "https://heibox.uni-heidelberg.de/f/d835d5b6ceda4d3aa910/?dl=1"
#     self.idx2syn = os.path.join(self.root, "index_synset.yaml")
#     if (not os.path.exists(self.idx2syn)):
#       download(URL, self.idx2syn)

#   def _prepare_human_to_integer_label(self):
#     self.human2integer = os.path.join(
#         self.data_specific, "sixray10_clsidx_to_labels.txt")

#     int2lbl = {
#         0: 'Gun',
#         1: 'Knife',
#         2: 'Wrench',
#         3: 'Pliers',
#         4: 'Scissors'
#     }

#     if (not os.path.exists(self.human2integer)):
#       with open(self.human2integer, "w") as f:
#         for key, value in int2lbl.items():
#           f.write(f"{key}: '{value}'\n")

#   def _load(self):
#     with open(self.txt_filelist, "r") as f:
#       self.relpaths = f.read().splitlines()
#       l1 = len(self.relpaths)
#       # self.relpaths = self._filter_relpaths(self.relpaths)
#       # print("Removed {} files from filelist during filtering.".format(
#       #     l1 - len(self.relpaths)))

#     self.synsets = [p.split("/")[0] for p in self.relpaths]
#     self.abspaths = [os.path.join(self.datadir, p) for p in self.relpaths]

#     unique_synsets = np.unique(self.synsets)
#     class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
#     if not self.keep_orig_class_label:
#       self.class_labels = [class_dict[s] for s in self.synsets]
#     else:
#       self.class_labels = [self.synset2idx[s] for s in self.synsets]

#     with open(self.human_dict, "r") as f:
#       human_dict = f.read().splitlines()
#       human_dict = dict(line.split(maxsplit=1) for line in human_dict)

#     self.human_labels = [
#         human_dict[s.replace(".jpg", "")] for s in self.synsets]

#     labels = {
#         "relpath": np.array(self.relpaths),
#         "synsets": np.array(self.synsets),
#         "class_label": np.array(self.class_labels),
#         "human_label": np.array(self.human_labels),
#     }

#     if self.process_images:
#       self.size = retrieve(self.config, "size", default=256)
#       self.data = ImagePaths(self.abspaths,
#                              labels=labels,
#                              size=self.size,
#                              random_crop=self.random_crop,
#                              )
#     else:
#       self.data = self.abspaths


# class SixRayTrain(SixRayBase):
#   NAME = "SixRay_train"
#   # URL = "http://www.image-net.org/challenges/LSVRC/2012/"
#   # AT_HASH = "a306397ccf9c2ead27155983c254227c0fd938e2"
#   # FILES = [
#   #     "ILSVRC2012_img_train.tar",
#   # ]
#   SIZES = [
#       74960,
#   ]

#   def __init__(self, process_images=True, data_root=None, **kwargs):
#     self.process_images = process_images
#     self.data_root = data_root
#     super().__init__(**kwargs)

#   def _prepare(self):
#     if self.data_root:
#       self.root = self.data_root
#     else:
#       cachedir = os.environ.get(
#           "XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
#       self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)

#     self.datadir = os.path.join(self.root, "JPEGImage")
#     self.expected_length = 74960
#     self.random_crop = retrieve(self.config, "SixRayTrain/random_crop",
#                                 default=True)

#     self.data_specific = os.path.join(self.root, "ImageSet", "10", self.NAME)

#     self.txt_filelist = os.path.join(self.data_specific, "train.txt")

#     if not tdu.is_prepared(self.data_specific):
#       # prep
#       # if not os.path.exists(self.txt_filelist):
#       print("Preparing dataset {} in {}".format(self.NAME, self.data_specific))

#       datadir = self.datadir

#       df = pd.read_csv(os.path.join(self.data_specific, "train.csv"))
#       file_list = df["name"].to_list()

#       available_images = os.listdir(self.datadir)
#       available_images = [i.replace(".jpg", "") for i in available_images]
#       filelist = [
#           datadir + f"/{name}.jpg" for name in file_list if name in available_images]

#       filelist = [os.path.relpath(p, start=datadir) for p in filelist]
#       filelist = sorted(filelist)
#       filelist = "\n".join(filelist)+"\n"
#       with open(self.txt_filelist, "w") as f:
#         f.write(filelist)

#       tdu.mark_prepared(self.data_specific)


# class SixRayValidation(SixRayBase):
#   NAME = "SixRay_validation"
#   # URL = "http://www.image-net.org/challenges/LSVRC/2012/"
#   # AT_HASH = "5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5"
#   # VS_URL = "https://heibox.uni-heidelberg.de/f/3e0f6e9c624e45f2bd73/?dl=1"
#   # FILES = [
#   #     "ILSVRC2012_img_val.tar",
#   #     "validation_synset.txt",
#   # ]
#   SIZES = [
#       13412,
#   ]

#   def __init__(self, process_images=True, data_root=None, **kwargs):
#     self.data_root = data_root
#     self.process_images = process_images
#     super().__init__(**kwargs)

#   def _prepare(self):
#     if self.data_root:
#       self.root = self.data_root
#     else:
#       cachedir = os.environ.get(
#           "XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
#       self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)

#     self.datadir = os.path.join(self.root, "JPEGImage")
#     self.expected_length = 13412
#     self.random_crop = retrieve(self.config, "SixRayValidation/random_crop",
#                                 default=False)

#     self.data_specific = os.path.join(self.root, "ImageSet", "10", self.NAME)

#     self.txt_filelist = os.path.join(self.data_specific, "test.txt")

#     # if not tdu.is_prepared(self.root):
#     if not tdu.is_prepared(self.data_specific):

#       # prep
#       print("Preparing dataset {} in {}".format(self.NAME, self.root))

#       datadir = self.datadir

#       df = pd.read_csv(os.path.join(self.data_specific, "test.csv"))
#       file_list = df["name"].to_list()

#       available_images = os.listdir(self.datadir)
#       available_images = [i.replace(".jpg", "") for i in available_images]
#       filelist = [
#           datadir + f"/{name}.jpg" for name in file_list if name in available_images]

#       # filelist = [datadir + f"/{name}.jpg" for name in file_list]
#       filelist = [os.path.relpath(p, start=datadir) for p in filelist]
#       filelist = sorted(filelist)
#       filelist = "\n".join(filelist)+"\n"
#       with open(self.txt_filelist, "w") as f:
#         f.write(filelist)

#       tdu.mark_prepared(self.data_specific)


# class SixRay(Dataset):
#   def __init__(self, size=None,
#                degradation=None, downscale_f=4, min_crop_f=0.5, max_crop_f=1.,
#                random_crop=True):
#     """
#     SixRay Dataloader
#     Performs following ops in order:
#     1.  crops a crop of size s from image either as random or center crop
#     2.  resizes crop to size with cv2.area_interpolation
#     3.  degrades resized crop with degradation_fn

#     :param size: resizing to size after cropping
#     :param degradation: degradation_fn, e.g. cv_bicubic or bsrgan_light
#     :param downscale_f: Low Resolution Downsample factor
#     :param min_crop_f: determines crop size s,
#       where s = c * min_img_side_len with c sampled from interval (min_crop_f, max_crop_f)
#     :param max_crop_f: ""
#     :param data_root:
#     :param random_crop:
#     """
#     self.base = self.get_base()
#     assert size
#     assert (size / downscale_f).is_integer()
#     self.size = size
#     self.LR_size = int(size / downscale_f)
#     self.min_crop_f = min_crop_f
#     self.max_crop_f = max_crop_f
#     assert (max_crop_f <= 1.)
#     self.center_crop = not random_crop

#     self.image_rescaler = albumentations.SmallestMaxSize(
#         max_size=size, interpolation=cv2.INTER_AREA)

#     # gets reset later if incase interp_op is from pillow
#     self.pil_interpolation = False

#     if degradation == "bsrgan":
#       self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)

#     elif degradation == "bsrgan_light":
#       self.degradation_process = partial(
#           degradation_fn_bsr_light, sf=downscale_f)

#     else:
#       interpolation_fn = {
#           "cv_nearest": cv2.INTER_NEAREST,
#           "cv_bilinear": cv2.INTER_LINEAR,
#           "cv_bicubic": cv2.INTER_CUBIC,
#           "cv_area": cv2.INTER_AREA,
#           "cv_lanczos": cv2.INTER_LANCZOS4,
#           "pil_nearest": PIL.Image.NEAREST,
#           "pil_bilinear": PIL.Image.BILINEAR,
#           "pil_bicubic": PIL.Image.BICUBIC,
#           "pil_box": PIL.Image.BOX,
#           "pil_hamming": PIL.Image.HAMMING,
#           "pil_lanczos": PIL.Image.LANCZOS,
#       }[degradation]

#       self.pil_interpolation = degradation.startswith("pil_")

#       if self.pil_interpolation:
#         self.degradation_process = partial(
#             TF.resize, size=self.LR_size, interpolation=interpolation_fn)

#       else:
#         self.degradation_process = albumentations.SmallestMaxSize(max_size=self.LR_size,
#                                                                   interpolation=interpolation_fn)

#   def __len__(self):
#     return len(self.base)

#   def __getitem__(self, i):
#     example = self.base[i]
#     image = Image.open(example["file_path_"])

#     if not image.mode == "RGB":
#       image = image.convert("RGB")

#     image = np.array(image).astype(np.uint8)

#     min_side_len = min(image.shape[:2])
#     crop_side_len = min_side_len * \
#         np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
#     crop_side_len = int(crop_side_len)

#     if self.center_crop:
#       self.cropper = albumentations.CenterCrop(
#           height=crop_side_len, width=crop_side_len)

#     else:
#       self.cropper = albumentations.RandomCrop(
#           height=crop_side_len, width=crop_side_len)

#     image = self.cropper(image=image)["image"]
#     image = self.image_rescaler(image=image)["image"]

#     if self.pil_interpolation:
#       image_pil = PIL.Image.fromarray(image)
#       LR_image = self.degradation_process(image_pil)
#       LR_image = np.array(LR_image).astype(np.uint8)

#     else:
#       LR_image = self.degradation_process(image=image)["image"]

#     example["image"] = (image/127.5 - 1.0).astype(np.float32)
#     example["LR_image"] = (LR_image/127.5 - 1.0).astype(np.float32)

#     return example


# class SixRayTrainD(SixRay):
#   def __init__(self, **kwargs):
#     super().__init__(**kwargs)

#   def get_base(self):

#     dset = SixRayTrain(process_images=False,
#                        data_root=f"/home/{pwd.getpwuid(os.getuid())[0]}/datasets/sixray", is_train=True)
#     return dset
#     # return Subset(dset)


# class SixRayValidationD(SixRay):
#   def __init__(self, **kwargs):
#     super().__init__(**kwargs)

#   def get_base(self):
#     dset = SixRayValidation(process_images=False,
#                             data_root=f"/home/{pwd.getpwuid(os.getuid())[0]}/datasets/sixray", is_train=False)  
#     return dset
#     # return Subset(dset)





class SixRay(Dataset):
  def __init__(self, csv_path, root_dir,degradation=None, min_crop_f=0.5, max_crop_f=1.,
               random_crop=True, downscale_f=4, size=256, data_type='train', transform=None):
    self.csv_path = csv_path
    self.root_dir = root_dir
    self.data_type = data_type
    self.transforms = transforms

    self.data = pd.read_csv(csv_path)
    self.data = self.data[self.data['subset'] == data_type]

    self.classes = ["Gun", "Knife", "Wrench", "Pliers", "Scissors", "Negative"]
    self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}    
    self.idx_to_human = {idx: cls_name for idx, cls_name in enumerate(self.classes)}

    if not transform:
      self.transform = transforms.Compose([transforms.Resize((size, size)),
                                      transforms.ToTensor()])
                              
    self.size = size
    self.LR_size = int(size / downscale_f)
    self.min_crop_f = min_crop_f
    self.max_crop_f = max_crop_f
    assert (max_crop_f <= 1.)
    self.center_crop = not random_crop

    self.image_rescaler = albumentations.SmallestMaxSize(
        max_size=size, interpolation=cv2.INTER_AREA)

    # gets reset later if incase interp_op is from pillow
    self.pil_interpolation = False

    if degradation == "bsrgan":
      self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)

    elif degradation == "bsrgan_light":
      self.degradation_process = partial(
          degradation_fn_bsr_light, sf=downscale_f)

    else:
      interpolation_fn = {
          "cv_nearest": cv2.INTER_NEAREST,
          "cv_bilinear": cv2.INTER_LINEAR,
          "cv_bicubic": cv2.INTER_CUBIC,
          "cv_area": cv2.INTER_AREA,
          "cv_lanczos": cv2.INTER_LANCZOS4,
          "pil_nearest": PIL.Image.NEAREST,
          "pil_bilinear": PIL.Image.BILINEAR,
          "pil_bicubic": PIL.Image.BICUBIC,
          "pil_box": PIL.Image.BOX,
          "pil_hamming": PIL.Image.HAMMING,
          "pil_lanczos": PIL.Image.LANCZOS,
      }[degradation]

      self.pil_interpolation = degradation.startswith("pil_")

      if self.pil_interpolation:
        self.degradation_process = partial(
            TF.resize, size=self.LR_size, interpolation=interpolation_fn)

      else:
        self.degradation_process = albumentations.SmallestMaxSize(max_size=self.LR_size,
                                                                  interpolation=interpolation_fn)

  def get_num_classes(self):
    return len(self.classes)

  def __getitem__(self, idx):
    img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
    image = Image.open(img_path)

    label = self.data.iloc[idx, 2]
    label = torch.tensor(self.class_to_idx[label], dtype=torch.long)

    if not image.mode == "RGB":
      image = image.convert("RGB")

    image = np.array(image).astype(np.uint8)

    min_side_len = min(image.shape[:2])
    crop_side_len = min_side_len * \
        np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
    crop_side_len = int(crop_side_len)

    if self.center_crop:
      self.cropper = albumentations.CenterCrop(
          height=crop_side_len, width=crop_side_len)

    else:
      self.cropper = albumentations.RandomCrop(
          height=crop_side_len, width=crop_side_len)

    image = self.cropper(image=image)["image"]
    image = self.image_rescaler(image=image)["image"]

    if self.pil_interpolation:
      image_pil = PIL.Image.fromarray(image)
      LR_image = self.degradation_process(image_pil)
      LR_image = np.array(LR_image).astype(np.uint8)

    else:
      LR_image = self.degradation_process(image=image)["image"]

    example = {}

    example["image"] = (image/127.5 - 1.0).astype(np.float32)
    example["LR_image"] = (LR_image/127.5 - 1.0).astype(np.float32)
    example["class_label"] = label
    example["human_label"] = self.idx_to_human[label]
    

    return example

  def __len__(self):
    return len(self.data)
