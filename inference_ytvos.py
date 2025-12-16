'''
Inference code for SgMg, on refer_youtube_vos
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import util.misc as utils
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image, ImageDraw
import math
import torch.nn.functional as F
import json

import opts
from tqdm import tqdm

import multiprocessing as mp
import threading
import warnings
warnings.filterwarnings("ignore")

from utils import colormap
from torch.cuda.amp import autocast
import time


# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()

# build transform
transform = T.Compose([
    T.Resize(360),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main(args):
    args.dataset_file = "ytvos"
    args.masks = True
    args.batch_size == 1
    args.eval = True

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    split = args.split
    # save path
    output_dir = args.output_dir
    save_path_prefix = os.path.join(output_dir, "Annotations")
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    save_visualize_path_prefix = os.path.join(output_dir, split + '_images')
    if args.visualize:
        if not os.path.exists(save_visualize_path_prefix):
            os.makedirs(save_visualize_path_prefix)

    # load data
    root = Path(args.ytvos_path) # data/refer_youtube_vos
    img_folder = os.path.join(root, split, "JPEGImages")
    flow_folder = os.path.join(root, split, "JPEGDepth")
    meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    valid_test_videos = set(data.keys())
    # for some reasons the competition's validation expressions dict contains both the validation (202) &
    # test videos (305). so we simply load the test expressions dict and use it to filter out the test videos from
    # the validation expressions dict:
    # test_meta_file = os.path.join(root, "meta_expressions", "test", "meta_expressions.json")
    # with open(test_meta_file, 'r') as f:
    #     test_data = json.load(f)['videos']
    # test_videos = set(test_data.keys())
    # valid_videos = valid_test_videos - test_videos
    # video_list = sorted([video for video in valid_videos])
    # assert len(video_list) == 202, 'error: incorrect number of validation videos'

    # create subprocess
    thread_num = args.ngpu
    global result_dict
    result_dict = mp.Manager().dict()

    processes = []
    lock = threading.Lock()
    video_list = sorted([video for video in valid_test_videos])
    video_num = len(video_list)
    print("Total video num is {}.".format(video_num))
    per_thread_video_num = video_num // thread_num

    start_time = time.time()
    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        p = mp.Process(target=sub_processor, args=(lock, i, args, data,
                                                   save_path_prefix, save_visualize_path_prefix,
                                                   img_folder, flow_folder, sub_video_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time

    result_dict = dict(result_dict)
    num_all_frames_gpus = 0
    for pid, num_all_frames in result_dict.items():
        num_all_frames_gpus += num_all_frames

    print("Total inference time: %.4f s" %(total_time))


def sub_processor(lock, pid, args, data, save_path_prefix, save_visualize_path_prefix, img_folder, flow_folder,
                  video_list):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
    torch.cuda.set_device(pid)

    # model
    model, criterion, _ = build_model(args)
    device = args.device
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if pid == 0:
        print('number of params:', n_parameters)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if 'args' in checkpoint:
            print("Loaded Checkpoint args: {}".format(checkpoint['args']))
        del checkpoint
    else:
        raise ValueError('Please specify the checkpoint for inference.')


    num_all_frames = 0
    model.eval()

    # Step 1: Register forward hook
    activation = {}
    def get_activation(name, mod, inp, out):
        activation[name] = out
    model.modality_fusion.register_forward_hook(get_activation)



    for idx_, video in enumerate(video_list):
        # if idx_ < 47:
        #     continue
        torch.cuda.empty_cache()
        metas = []
        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        video_len = len(data[video]["frames"])

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas

        # store images
        frames = data[video]["frames"]
        video_name = video
        imgs = []
        flows = []
        for t in range(video_len):
            frame = frames[t]
            img_path = os.path.join(img_folder, video_name, frame + ".jpg")
            flow_path = os.path.join(flow_folder, video_name, frame + ".png")
            img = Image.open(img_path).convert('RGB')
            flow = Image.open(flow_path).convert('RGB')
            origin_w, origin_h = img.size
            imgs.append(transform(img))  # list[img]
            flows.append(transform(flow))  # list[img]

        imgs = torch.stack(imgs, dim=0).to(args.device)  # [video_len, 3, h, w]
        flows = torch.stack(flows, dim=0).to(args.device)  # [video_len, 3, h, w]
        img_h, img_w = imgs.shape[-2:]
        size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
        target = {"size": size}

        # 2. For each expression
        for i in range(num_expressions):
            video_name = meta[i]["video"]
            exp = meta[i]["exp"]
            exp_id = meta[i]["exp_id"]
            frames = meta[i]["frames"]
            print(f'video name:{video_name}, expression: {exp}, video len: {len(frames)}')

            video_len = len(frames)
            with torch.no_grad():
                with autocast(args.amp):
                    outputs = model([imgs], [flows], [exp], [target])



            # Step 2: Access activation and generate heatmap
                features = activation['modality_fusion']  # Shape: [video_len, C, h', w']
                print('feature shape',features.shape)
            for t in range(video_len):
                feature_map = features[t].unsqueeze(0)  # Shape: [1, C, h', w']
                heatmap = feature_map.mean(dim=1).squeeze()  # Shape: [h', w']
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                heatmap = F.interpolate(heatmap.unsqueeze(0).unsqueeze(0), size=(origin_h, origin_w), mode='bilinear', align_corners=False).squeeze()


                # Save the heatmap as grayscale image
                heatmap_save_path_prefix='lyf/SgMg-main/heatmap'
                heatmap_np = (heatmap.cpu().numpy() * 255).astype(np.uint8)
                heatmap_save_path = os.path.join(heatmap_save_path_prefix, video_name, exp_id)
                if not os.path.exists(heatmap_save_path):
                    os.makedirs(heatmap_save_path)
                heatmap_file = os.path.join(heatmap_save_path, frames[t] + "_heatmap.png")
                cv2.imwrite(heatmap_file, heatmap_np)

                # Step 3: Denormalize the original image
                img = imgs[t].clone()
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                img = img.permute(1,2,0).cpu().numpy()
                img = (img * 255).astype(np.uint8)

                # Step 4: Overlay heatmap on original image
                heatmap_colored = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img, 0.5, heatmap_colored, 0.5, 0)

                # Step 5: Save the visualization
                save_visualize_path = os.path.join(save_visualize_path_prefix, video_name, exp_id)
                if not os.path.exists(save_visualize_path):
                    os.makedirs(save_visualize_path)
                save_file = os.path.join(save_visualize_path, frames[t] + ".png")
                cv2.imwrite(save_file, overlay)



        with lock:
            progress.update(1)

    result_dict[str(pid)] = num_all_frames
    with lock:
        progress.close()


# visuaize functions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# Visualization functions
def draw_reference_points(draw, reference_points, img_size, color):
    W, H = img_size
    for i, ref_point in enumerate(reference_points):
        init_x, init_y = ref_point
        x, y = W * init_x, H * init_y
        cur_color = color
        draw.line((x-10, y, x+10, y), tuple(cur_color), width=4)
        draw.line((x, y-10, x, y+10), tuple(cur_color), width=4)

def draw_sample_points(draw, sample_points, img_size, color_list):
    alpha = 255
    for i, samples in enumerate(sample_points):
        for sample in samples:
            x, y = sample
            cur_color = color_list[i % len(color_list)][::-1]
            cur_color += [alpha]
            draw.ellipse((x-2, y-2, x+2, y+2), 
                            fill=tuple(cur_color), outline=tuple(cur_color), width=1)

def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8') # np
    mask = mask > 0.5

    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img = Image.fromarray(origin_img)
    return origin_img

  

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SgMg inference script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    main(args)
    print("Save results at: {}.".format(os.path.join(args.output_dir, "Annotations")))
