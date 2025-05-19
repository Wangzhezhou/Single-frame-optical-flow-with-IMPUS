import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

# test our predicted sintel (500)
# def demo(args):
#     model = torch.nn.DataParallel(RAFT(args))
#     model.load_state_dict(torch.load(args.model))
#     model = model.module.to(DEVICE)
#     model.eval()
    
#     image_dir = 'sintel_result/kitti_3'
#     base_image_path = os.path.join(image_dir, '000002_10.png')
#     output_file = 'optical_flow_result/KITTI_flow_res/kitti_03.npy'
    
#     with torch.no_grad():
#         images = [os.path.join(image_dir, f"frame_{i:03d}.png") for i in range(1, 501)]
#         flow_results = []
        
#         for imfile2 in images:
#             image1 = load_image(base_image_path)
#             if not os.path.exists(imfile2):
#                 print('image2 not exist.')
#                 continue
            
#             image2 = load_image(imfile2)
#             print(f"Image1 shape: {image1.shape}, Image2 shape: {image2.shape}")

#             padder = InputPadder(image1.shape)
#             image1, image2 = padder.pad(image1, image2)
            
#             flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
#             flow_unpadded = padder.unpad(flow_up).squeeze(0)
#             flow_results.append(flow_unpadded.cpu().numpy())
        
#         flow_results = np.stack(flow_results, axis=0)  # (500, 2, h, w)
#         np.save(output_file, flow_results)

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module.to(DEVICE)
    model.eval()
    
    image_dir = 'sintel_result/sintel_svd'
    base_image_path = os.path.join(image_dir, 'org_11.png')
    imfile2 = os.path.join(image_dir, '11.png')
    output_file = 'optical_flow_result/Sintel_svd/sintel_svd_11.npy'
    
    with torch.no_grad():
        image1 = load_image(base_image_path)
        image2 = load_image(imfile2)
        print(f"Image1 shape: {image1.shape}, Image2 shape: {image2.shape}")
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        flow_unpadded = padder.unpad(flow_up)
        np.save(output_file, flow_unpadded.cpu().numpy())
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
