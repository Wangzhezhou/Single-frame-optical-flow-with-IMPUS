# _ProbDiffFlow_: An Efficient Learning-Free Framework for Probabilistic Single-Image Optical Flow Estimation

This repository contains the source code for our paper:

[ProbDiffFlow: An Efficient Learning-Free Framework for Probabilistic Single-Image Optical Flow Estimation](https://arxiv.org/abs/2503.12348)


![Overall_Framework](https://github.com/Wangzhezhou/Single-frame-optical-flow-with-IMPUS/blob/main/Images/overall_framework.jpg)

## Step 1: Sample Nearby (Second) Images

### Requirements:

The code has been tested on PyTorch 2.1.2, Python 3.10, and CUDA 11.8.

`pip install -r requirements.txt`

### Dataset Configuration

This project uses three benchmark datasets:
- [KITTI](https://www.cvlibs.net/datasets/kitti/)
- [Sintel](https://sintel.is.tue.mpg.de/)
- [Spring](https://spring-benchmark.org)

The main script for testing is `IMPUS_test.ipynb`, which is designed for single-image input. To run a test, set the variable `'input_image_1'` to the path of your test image. The prompt used in the testing process can be generated using **GPT-4o (ChatGPT)**.

**Note:** `IMPUS_test.ipynb` generates the sampled results in `.npy` format, containing the latent vectors or intermediate representations.

## Step 2: Optical Flow Distribution Estimation

Here, we use a pre-trained [RAFT](https://github.com/princeton-vl/RAFT) model to estimate the optical flow between the original image and each of the 500 generated images.

### Modified RAFT Files

To adapt RAFT to our pipeline, we made modifications to the following files:
- demo.py
- core/utils/flow_vis.py
- core/utils/frame_utils.py
- core/utils/utils.py
- core/datasets.py

The modified files are placed in the `RAFT/` folder of this project. Make sure to **replace the corresponding files** in the original RAFT project directory before running.

### Running

To compute the optical flow, run the following command:

`python demo.py --model models/raft-sintel.pth --path image_dir --mixed_precision`

**Note**: `demo.py` produces the estimated optical flow results in `.flo` format for each image pair.

## Step 3: Flow Distribution Analysis

The analysis of the predicted flow distribution is performed in the `flow_analysis.ipynb`.

This includes:
- Computing evaluation metrics such as **F1-all**, **EPE (Endpoint Error)**, etc.
- Visualizing the optical flow fields and their distributions

Make sure to run this notebook after generating `.flo` files from Step 2.

