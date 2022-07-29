# Infrared Image Colorization
By Hyeongyu Kim

Department of Electrical and Computer Engineering at SungKyunKwan University in the M.S Course.

This is a Pytorch implementation of Infrared Image Colorization Using Neural Networks. This repository includes the implementation of ["Learn to 
Recover Visible Color for Video Surveillance in a Day"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460477.pdf) as well, so that you can train and compare among base CNN model.

## Abstract
In the conventional near-infrared (NIR) image colorization methods, color consistency is reduced, texture is damaged, and artifacts occur. In particular, in the method using Generative Adversarial Networks (GAN), the quantitative value is calculated high, but the problem of image distortion occurs. To solve this problem, we propose a feature fusion module using a multi-scale channel attention module (MS-CAM) that extracts edges using Canny filter and fuses two color images with feature maps. The proposed method improves the PSNR value of NIR2VC compared to the existing method in the VSIAD (Video surveillance in a day) dataset with two types of data, VNIR2VC and NIR2VC. This satisfies the purpose required in this paper and contributes to performance improvement. Qualitatively, the result has an excellent ability to capture the consistency of the building exterior wall color and texture components such as wood, which has an excellent effect in urban areas with dense buildings or natural images. In addition, the proposed method can also be applied to the field of nighttime activities and monitoring such as CCTV and military facilities, as well as thermal imaging cameras with properties similar to near-infrared rays. For future research, research on artifact improvement methods, performance improvement using GAN and model compression should be conducted.

## Baseline
![Baseline structure](./images/Baseline_structure.jpg)
G. Wu proposed State Synchronization Networks (SSN) for NIR image colorization in 2020 ECCV, ["Learn to Recover Visible Color for Video Surveillance in a Day"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460477.pdf). This structure uses one encoder and two parallel decoder and State Synchronization Module (SSM). encoder follows [ResNet](https://arxiv.org/abs/1512.03385?context=cs) using several residual blocks and two decoders have the same structure except for the final prediction layer. In addition, the deconvolutional layer and skip connection used in [U-Net](https://arxiv.org/abs/1505.04597) are applied to connect them with the same height and width. 

## Proposed Method
We proposed a method for enhancing the edge, texture and color consistency. This structure shows a proposed method for edge enhancement and a feature fusion module. It is composed a three identical baseline structures and feature fusion modules.
First, we extract the feature maps from input image and edge. Second, the previously output colorized images extracted are fused using the feature fusion module.
Feature fusion module utilizes [MS-CAM](https://arxiv.org/abs/2009.14082) twice. 
By separating the Lab image, the L and ab are fused independently. The colorized images from input and edges are different, but they are similar to each other. When two images are concatenated in channels, information gets mixed up and leads to poor performance. Since the L includes brightness and the ab includes chrominance, if different components are used as it is, it is also mixed without being refined. In order to prevent this, the L and ab are separated and independently purified. Finally, another baseline structure takes a fused colorized images and provides the outputs.

## Experiments
In VSIAD dataset, separated into two parts, i.e., paired VNIR images of daytime and NIR images of night-time. We split 20,000 image pairs into train, validation and test as 3:1:1. We select a batch size of 40 and randomly crop 256 × 256 patches from 640 × 480 image of full resolution. The network is trained with a NVIDIA Tesla V100 and for 100,000 iterations. All of parameters are optimized by the Adam optimizer using initial learning late = 1e^-4, $\Beta$_1 = 0.9, $\Beta$_2 = 0.999, $\Epsilon$ = 1e^-8.
|Method|PSNR|SSIM|LPIPS|
|------|------|------|------|
|pix2pix|14.485|0.599|0.166|
|pix2pixHD|19.654|0.641|0.287|
|baseline|19.690|0.698|0.248|
|Proposed Method|20.479|0.688|0.255|
