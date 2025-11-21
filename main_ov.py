# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import json
import os
import copy
from PIL import Image
import psutil

from pathlib import Path

from timm.models import create_model
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms

import utils
import models.convnext
import models.convnext_isotropic

import openvino as ov
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm, ColorFormat
from openvino import Core, Model, set_batch, Layout, Type, Tensor, AsyncInferQueue, InferRequest
from openvino.preprocess.torchvision import PreprocessConverter

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', '-b', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--model', '-m', default='convnext_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', '-s', default=224, type=int, help='image input size')
    parser.add_argument('--use_amp', '-a', type=str2bool, default=False, 
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")
    parser.add_argument('--checkpoint', '-c', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--nb_classes', '-n', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--loop', '-l', type=int, default=20, help='Number of classes')
    parser.add_argument('--warmup', '-w', type=int, default=2, help='Number of classes')
    
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--crop_pct', type=float, default=None)

    parser.add_argument('--nstreams', '-t', type=int, default=2, help='Number of ov streams')
    return parser

def build_transform(args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.ConvertImageDtype(torch.float32))
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def load_torch_model(model_name: str, checkpoint_path: str) :
    model = create_model(model_name, 
        pretrained=True,
        num_classes=args.nb_classes, 
        )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    
    ov_path = Path(checkpoint_path).with_suffix('.xml')
    if not ov_path.exists() :
        ov_model = ov.convert_model(model, example_input=torch.rand(1, 3, args.input_size, args.input_size))
        ov.save_model(ov_model, ov_path, compress_to_fp16=False)
        print(f"OpenVINO model is saved to {ov_path}")

    return model

def prepare_input_torch(images, batch_size, transform):
    batches = []
    for i in range(0, len(images), batch_size):
        output = []
        for j in range(batch_size):
            image = transform(images[i+j])
            output.append(image)
        output = torch.stack(output, dim=0)
        batches.append(output)
    return batches

def prepare_input_torch1(image, batch_size, transform):
    images = []
    for i in range(batch_size) :
        images.append(transform(image))
    # images = transform(image).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return torch.stack(images, dim=0)

@torch.no_grad()
def inference_torch(name, in_files, model, transform, batch_size, warmup, loop, use_amp=False):
    # batched_images = prepare_input_torch(in_files, batch_size, transform)
    for i in range(warmup):
        batched_images = prepare_input_torch1(in_files[i], batch_size, transform)
        if use_amp:
            with torch.amp.autocast('cpu', dtype=torch.bfloat16) :
                output = model(batched_images).cpu().float().numpy()
        else:
            output = model(batched_images).cpu().float().numpy()
    
    st = time.perf_counter()
    for i in range(loop):
        batched_images = prepare_input_torch1(in_files[i], batch_size, transform)
        if use_amp:
            with torch.amp.autocast('cpu', dtype=torch.bfloat16) :
                output = model(batched_images).cpu().float().numpy()
        else:
            output = model(batched_images).cpu().float().numpy()
    et = time.perf_counter()
    total_latency = et - st
    print(f'{name} {loop} runs (use_amp={use_amp}) use {total_latency:.4f}, Average inference time: {total_latency/loop/batch_size:.4f} seconds, {batch_size*loop/total_latency:.4f} FPS')
    return output

def load_ov_model(model_path: str, nstreams: int, batchsize: int, amx: int):
    core = ov.Core()
    ov_model = core.read_model(model=model_path)

    ppp = PrePostProcessor(ov_model)
    ppp.input(0).tensor()\
            .set_element_type(Type.u8)\
            .set_layout(Layout('NHWC'))\
            .set_color_format(ColorFormat.RGB)

    ppp.input().model().set_layout(Layout('NCHW'))

    ppp.input(0).preprocess()\
        .resize(ResizeAlgorithm.RESIZE_BICUBIC_PILLOW, 256, 256)\
        .crop([0, 16, 16, 0], [batchsize, 240, 240, 3])\
        .convert_element_type(Type.f32)\
        .mean([123.675, 116.28, 103.53])\
        .scale([58.395, 57.12, 57.375])

    ov_model = ppp.build()
    
    # test_input = np.random.randint(255, size=(260, 260, 3), dtype=np.uint16)
    # ov_model = PreprocessConverter.from_torchvision(
    #     model=ov_model, transform=preprocess_pipeline, input_example=Image.fromarray(test_input.astype("uint8"), "RGB")
    # )


    data_type = 'bf16' if amx==1 else 'f16' if amx==2 else 'f32'
    hint = 'THROUGHPUT' if nstreams>1 else 'LATENCY'
    config = {}
    config['NUM_STREAMS'] = str(nstreams)
    config['PERF_COUNT'] = 'NO'
    config['INFERENCE_PRECISION_HINT'] = data_type
    config['PERFORMANCE_HINT'] = hint
    compiled_model = core.compile_model(ov_model, 'CPU', config)
    num_requests = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    infer_queue = AsyncInferQueue(compiled_model, num_requests)
    return compiled_model, infer_queue

def prepare_input_ov(images, batch_size):
    batches = []
    for i in range(0, len(images), batch_size):
        output = []
        for j in range(batch_size):
            image = np.asarray(images[i+j])
            output.append(image)
        output = np.stack(output, axis=0)
        batches.append(output)
    return batches

def prepare_input_ov1(image, batch_size):
    images = np.asarray(image)
    images = np.expand_dims(images, axis=0)
    images = np.repeat(images, repeats=batch_size, axis=0)
    return images

def inference_ov(name, in_files, model, batch_size, warmup, loop):
    # batched_images = prepare_input_ov(in_files, batch_size)
    for i in range(warmup):
        batched_images = prepare_input_ov1(in_files[i], batch_size)
        output = model(batched_images)[0]
  
    st = time.perf_counter()
    for i in range(loop):
        batched_images = prepare_input_ov1(in_files[i], batch_size)
        output = model(batched_images)[0]
    et = time.perf_counter()
    total_latency = et - st
    print(f'{name} {loop} runs use {total_latency:.4f}, Average inference time: {total_latency/loop/batch_size:.4f} seconds, {batch_size*loop/total_latency:.4f} FPS')
    return output

def inference_ov_async(name, in_files, infer_queue, batch_size, warmup, loop):
    # batched_images = prepare_input_ov(in_files, batch_size)
    ov_results = {}
    def completion_callback(infer_request: InferRequest, index: any) :
        ov_results[index] = copy.deepcopy(infer_request.get_output_tensor(0).data)
    infer_queue.set_callback(completion_callback)

    for i in range(warmup):
        batched_images = prepare_input_ov1(in_files[0], batch_size)
        infer_queue.start_async({0: batched_images}, userdata=i)#, share_inputs=True)
    infer_queue.wait_all()

    ov_results = {}
    st = time.perf_counter()
    for i in range(loop):
        batched_images = prepare_input_ov1(in_files[0], batch_size)
        infer_queue.start_async({0: batched_images}, userdata=i)#, share_inputs=True)
    infer_queue.wait_all()
    et = time.perf_counter()
    total_latency = et - st
    print(f'{name} {loop} runs use {total_latency:.4f}, Average inference time: {total_latency/loop/batch_size:.4f} seconds, {batch_size*loop/total_latency:.4f} FPS')
    return ov_results[0]

def load_ov_model1(model_path: str, transform, batchsize: int, amx: int):
    core = ov.Core()
    ov_model = core.read_model(model=model_path)

    stream_num=1
    data_type = 'bf16' if amx==1 else 'f16' if amx==2 else 'f32'
    hint = 'THROUGHPUT' if stream_num>1 else 'LATENCY'
    config = {}
    config['NUM_STREAMS'] = str(stream_num)
    config['PERF_COUNT'] = 'NO'
    config['INFERENCE_PRECISION_HINT'] = data_type
    config['PERFORMANCE_HINT'] = hint
    compiled_model = core.compile_model(ov_model, 'CPU', config)
    num_requests = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    infer_queue = AsyncInferQueue(compiled_model, num_requests)
    return compiled_model, infer_queue

def inference_ov1(name, in_files, model, transform, batch_size, warmup, loop):
    for i in range(warmup):
        batched_images = prepare_input_torch1(in_files[i], batch_size, transform)
        output = model(batched_images)[0]
  
    st = time.perf_counter()
    for i in range(loop):
        batched_images = prepare_input_torch1(in_files[i], batch_size, transform)
        output = model(batched_images)[0]
    et = time.perf_counter()
    total_latency = et - st
    print(f'{name} {loop} runs use {total_latency:.4f}, Average inference time: {total_latency/loop/batch_size:.4f} seconds, {batch_size*loop/total_latency:.4f} FPS')
    return output

def inference_ov_async1(name, in_files, infer_queue, transform, batch_size, warmup, loop):
    # batched_images = prepare_input_ov(in_files, batch_size)
    ov_results = {}
    def completion_callback(infer_request: InferRequest, index: any) :
        ov_results[index] = copy.deepcopy(infer_request.get_output_tensor(0).data)
    infer_queue.set_callback(completion_callback)

    for i in range(warmup):
        batched_images = prepare_input_torch1(in_files[i], batch_size, transform)
        infer_queue.start_async({0: batched_images}, userdata=i)#, share_inputs=True)
    infer_queue.wait_all()
   
    ov_results = {}
    st = time.perf_counter()
    for i in range(loop):
        batched_images = prepare_input_torch1(in_files[i], batch_size, transform)
        infer_queue.start_async({0: batched_images}, userdata=i)#, share_inputs=True)
    infer_queue.wait_all()
    et = time.perf_counter()
    total_latency = et - st
    print(f'{name} {loop} runs use {total_latency:.4f}, Average inference time: {total_latency/loop/batch_size:.4f} seconds, {batch_size*loop/total_latency:.4f} FPS')
    return ov_results[0]

def prepare_images(in_files, ncount) :
    image_list = []
    for i in range(ncount) :
        for filename in in_files:
            image = Image.open(filename).convert("RGB")
            image_list.append(image)
            if len(image_list) >= ncount:
                return image_list
    return image_list

def main(args):
    print(args)

    checkpoint = args.checkpoint
    model_name = args.model
    rtol = 1e-05
    atol = 1e-05
    
    if args.warmup > args.loop:
        args.loop = args.warmup
    # ncount = args.batch_size * args.nstreams * args.loop
    ncount = args.nstreams * args.loop
    images = prepare_images(args.input, ncount)
    
    transform = build_transform(args)
    # print("Transform = ")
    # if isinstance(transform, tuple):
    #     for trans in transform:
    #         print(" - - - - - - - - - - ")
    #         for t in trans.transforms:
    #             print(t)
    # else:
    #     for t in transform.transforms:
    #         print(t)
    # print("---------------------------")
    
    model = load_torch_model(model_name, checkpoint)
    output_torch_False = inference_torch("Torch_FP32", images, model, transform, args.batch_size, args.warmup, args.loop, False)
    output_torch_True = inference_torch("Torch_BF16", images, model, transform, args.batch_size, args.warmup, args.loop, True)

    allclose_False = np.allclose(output_torch_False, output_torch_True, rtol=rtol, atol=atol)
    diff_False = np.abs(output_torch_False - output_torch_True)
    print(f"Torch, 近似相等:{allclose_False}, max差异:{np.max(diff_False):.4f}, mean差异:{np.mean(diff_False):.4f}")

    ov_path = Path(checkpoint).with_suffix('.xml')
    mode_name = ['F32', 'BF16', 'F16']
    for i,name in enumerate(mode_name):            
        ov_model, infer_queue = load_ov_model(ov_path, args.nstreams, args.batch_size, amx=i)

        output_ov = inference_ov(f"OV_sync_{name}", images, ov_model, args.batch_size, args.warmup, args.loop)
        allclose_False = np.allclose(output_torch_False, output_ov, rtol=rtol, atol=atol)
        diff_False = np.abs(output_torch_False - output_ov)
        print(f"{name}, 近似相等_F32:{allclose_False}, max差异:{np.max(diff_False):.4f}, mean差异:{np.mean(diff_False):.4f}")
        allclose_True = np.allclose(output_torch_True, output_ov, rtol=rtol, atol=atol)
        diff_True = np.abs(output_torch_True - output_ov)
        print(f"{name}, 近似相等_BF16:{allclose_True}, max差异:{np.max(diff_True):.4f}, mean差异:{np.mean(diff_True):.4f}")

        output_ov = inference_ov_async(f"OV_async_{name}", images, infer_queue, args.batch_size, args.warmup, args.loop)
        allclose_False = np.allclose(output_torch_False, output_ov, rtol=rtol, atol=atol)
        diff_False = np.abs(output_torch_False - output_ov)
        print(f"{name}, 近似相等_F32:{allclose_False}, max差异:{np.max(diff_False):.4f}, mean差异:{np.mean(diff_False):.4f}")
        allclose_True = np.allclose(output_torch_True, output_ov, rtol=rtol, atol=atol)
        diff_True = np.abs(output_torch_True - output_ov)
        print(f"{name}, 近似相等_BF16:{allclose_True}, max差异:{np.max(diff_True):.4f}, mean差异:{np.mean(diff_True):.4f}")

        process = psutil.Process(os.getpid())        
        mem_info = process.memory_info()
        print(f"RSS: {mem_info.rss / 1024 ** 3:.2f} GB")  # 常驻内存
        print(f"VMS: {mem_info.vms / 1024 ** 3:.2f} GB")  # 虚拟内存
        del ov_model

    mode_name = ['F32', 'BF16', 'F16']
    for i,name in enumerate(mode_name):            
        ov_model, infer_queue = load_ov_model1(ov_path, args.nstreams, args.batch_size, amx=i)
        output_ov = inference_ov1(f"OV1_sync_{name}", images, ov_model, transform, args.batch_size, args.warmup, args.loop)
        allclose_False = np.allclose(output_torch_False, output_ov, rtol=rtol, atol=atol)
        diff_False = np.abs(output_torch_False - output_ov)
        print(f"{name}, 近似相等_F32:{allclose_False}, max差异:{np.max(diff_False):.4f}, mean差异:{np.mean(diff_False):.4f}")
        allclose_True = np.allclose(output_torch_True, output_ov, rtol=rtol, atol=atol)
        diff_True = np.abs(output_torch_True - output_ov)
        print(f"{name}, 近似相等_BF16:{allclose_True}, max差异:{np.max(diff_True):.4f}, mean差异:{np.mean(diff_True):.4f}")

        output_ov = inference_ov_async1(f"OV1_async_{name}", images, infer_queue, transform, args.batch_size, args.warmup, args.loop)
        allclose_False = np.allclose(output_torch_False, output_ov, rtol=rtol, atol=atol)
        diff_False = np.abs(output_torch_False - output_ov)
        print(f"{name}, 近似相等_F32:{allclose_False}, max差异:{np.max(diff_False):.4f}, mean差异:{np.mean(diff_False):.4f}")
        allclose_True = np.allclose(output_torch_True, output_ov, rtol=rtol, atol=atol)
        diff_True = np.abs(output_torch_True - output_ov)
        print(f"{name}, 近似相等_BF16:{allclose_True}, max差异:{np.max(diff_True):.4f}, mean差异:{np.mean(diff_True):.4f}")

        process = psutil.Process(os.getpid())        
        mem_info = process.memory_info()
        print(f"RSS: {mem_info.rss / 1024 ** 3:.2f} GB")  # 常驻内存
        print(f"VMS: {mem_info.vms / 1024 ** 3:.2f} GB")  # 虚拟内存
        del ov_model

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
