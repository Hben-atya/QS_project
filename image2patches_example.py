import torch
import os
import json
# from torch.utils.serialization import load_lua
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader

from util import *

from monai.transforms import (
    AddChanneld,
    LoadNifti,
    LoadNiftid,
    Orientationd,
    Rand3DElasticd,
    RandAffined,
    Spacingd,
    MapTransform,
    ToTensor

)



import tarfile

if fname.endswith("tar.gz"):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()
elif fname.endswith("tar"):
    tar = tarfile.open(fname, "r:")
    tar.extractall()
    tar.close()


# SetUp data directory:
root_dir = r'D:\MICA_BRaTS2018\Training'

json_file_path = os.path.join(root_dir, "dataset.json")
with open(json_file_path) as json_file:
    json_data = json.load(json_file)

train_data_dicts = json_data["training"]

# Load one image nifti file:

loader = LoadNifti(dtype=np.float32)
image, metadata = loader(train_data_dicts[0]["image"])
print(f"input: {train_data_dicts[0]['image']}")
print(f"image shape: {image.shape}")
print(f"image affine:\n{metadata['affine']}")
print(f"image pixdim:\n{metadata['pixdim']}")

patch_size = 15
batch_size = 1
stride = 14
# input is a batch for 2 encoders, 3D patch size
input_batch = torch.zeros(batch_size, 2, *([patch_size] *3))
# output is a batch from 3 decoders, 3D patch size
output_batch = torch.zeros(batch_size, 3, *([patch_size] *3))


moving_appear_trainset = np.expand_dims(image[0,...], axis=0)  # image size = (1, Z,Y,X)!
target_appear_trainset = np.expand_dims(image[1,...], axis=0)   # image size = (1, Z,Y,X)!

conv2tensor = ToTensor()
moving_appear_trainset = conv2tensor(moving_appear_trainset)
target_appear_trainset = conv2tensor(target_appear_trainset)
dataset_size = moving_appear_trainset.shape
# output: starting indices for patches in flattened image concattenated all images.
# input: batch,3D patch, XYZ size of image, equal stride at each direction
flat_idx = calculatePatchIdx3D(dataset_size[0], patch_size * torch.ones(3), dataset_size[1:],stride * torch.ones(3))
# flat_idx = calculatePatchIdx3D(batch_size, patch_size * torch.ones(3), dataset_size, stride * torch.ones(3))
flat_idx_select = torch.zeros(flat_idx.size())

for patch_idx in range(1, flat_idx.size()[0]):
    # a 4D position - batch,x,y,z from the flattened concattenated images
    patch_pos = idx2pos_4D(flat_idx[patch_idx].item(), dataset_size[1:])  # Todo: ask - the values are not int!!
    moving_patch = moving_appear_trainset[int(patch_pos[0]),
                   int(patch_pos[1]):int(patch_pos[1])+patch_size,
                   int(patch_pos[2]):int(patch_pos[2])+patch_size,
                   int(patch_pos[3]):int(patch_pos[3])+patch_size]
    target_patch = target_appear_trainset[int(patch_pos[0]),
                   int(patch_pos[1]):int(patch_pos[1])+patch_size,
                   int(patch_pos[2]):int(patch_pos[2])+patch_size,
                   int(patch_pos[3]):int(patch_pos[3])+patch_size]
    if (torch.sum(moving_patch) + torch.sum(target_patch) != 0):
        flat_idx_select[patch_idx] = 1

flat_idx_select = flat_idx_select.byte()
# will return indices where flat_idx_select is not zero
flat_idx = torch.masked_select(flat_idx, flat_idx_select)
# number of patches in one image?
N = flat_idx.size()[0] / batch_size


for iters in range(0, N):
    # vector in the size of batch size with values [0,flat_idx_size] will give us a random patch from the image
    train_idx = (torch.rand(batch_size).double() * flat_idx.size()[0])
    train_idx = torch.floor(train_idx).long()
    for slices in range(0, batch_size):
    # I think it will go through all patches in one image
        patch_pos = idx2pos_4D(flat_idx[train_idx[slices].item()].item(), dataset_size[1:])
        input_batch[slices, 0] = moving_appear_trainset[int(patch_pos[0]),
                                 int(patch_pos[1]):int(patch_pos[1])+patch_size,
                                 int(patch_pos[2]):int(patch_pos[2])+patch_size,
                                 int(patch_pos[3]):int(patch_pos[3])+patch_size]
        input_batch[slices, 1] = target_appear_trainset[int(patch_pos[0]),
                                 int(patch_pos[1]):int(patch_pos[1])+patch_size,
                                 int(patch_pos[2]):int(patch_pos[2])+patch_size,
                                 int(patch_pos[3]):int(patch_pos[3])+patch_size]
        # ground truth for each patch
        output_batch[slices] = train_m0[int(patch_pos[0]), :,
                               int(patch_pos[1]):int(patch_pos[1]) + patch_size,
                               int(patch_pos[2]):int(patch_pos[2]) + patch_size,
                               int(patch_pos[3]):int(patch_pos[3]) + patch_size]
