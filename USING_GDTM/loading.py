# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile

from mmtrack.core import results2outs
import numpy as np
import torch
import torchaudio
import cv2

@PIPELINES.register_module()
class DecodeJPEG(object):
    def __init__(self):
        pass

    def __call__(self, code):
        img = cv2.imdecode(code, 1)
        return img

# @PIPELINES.register_module()
# class LoadAudio(object):
#     def __init__(self):
#         self.spectro = torchaudio.transforms.Spectrogram()

#     def __call__(self, array):
#         array = torch.from_numpy(array)
#         array = array.unsqueeze(0)
#         sgram = self.spectro(array)
#         sgram = sgram.squeeze()
#         sgram = sgram.permute(1, 2, 0)
#         return sgram.numpy()
@PIPELINES.register_module()
class LoadAudio(object):
    def __init__(self, n_fft=400):
        self.spectro = torchaudio.transforms.Spectrogram(n_fft=n_fft)

    def __call__(self, array):
        array = array[:, 1:5]
        array = torch.from_numpy(array)
        array = array.unsqueeze(0)
        array = array.permute(0, 2, 1)
        sgram = self.spectro(array)
        sgram = sgram.permute(0, 2, 3, 1).squeeze()
        sgram = sgram.numpy()
        results = {
            'img': sgram, 
            'img_shape': sgram.shape,
            'ori_shape': sgram.shape, 
            'img_fields': ['img'],
            'filename': 'placeholder.jpg',
            'ori_filename': 'placeholder.jpg'
        }
        return results

@PIPELINES.register_module()
class LoadFromNumpyArray(object):
    def __init__(self, force_float32=False, transpose=False, force_rgb=False,
            remove_first_last=False):
        self.force_float32 = force_float32
        self.transpose = transpose
        self.force_rgb = force_rgb
        self.remove_first_last = remove_first_last

    def __call__(self, array):
        if self.remove_first_last:
            array = array[:, 1:5]
            if len(array) != 1056:
                num_zeros = 1056 - len(array)
                zeros = np.zeros([num_zeros, 4])
                array = np.concatenate([array, zeros], axis=0)
            #array = array.T
            array = array[:, np.newaxis, :]
        if self.force_float32:
            array = array.astype(np.float32)
        if self.transpose:
            array = array.T
        if len(array.shape) == 2: #add channel dimesion
            array = array[:, :, np.newaxis]
        if self.force_rgb:
            array = np.concatenate([array, array, array], axis=-1)
        array = np.nan_to_num(array, nan=0.0)
        results = {
            'img': array, 
            'img_shape': array.shape,
            'ori_shape': array.shape, 
            'img_fields': ['img'],
            'filename': 'placeholder.jpg',
            'ori_filename': 'placeholder.jpg'
        }
        return results

