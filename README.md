# Gap-junction-filter
# [Retinal Gap Junctions Convert Noise Distribution and Facilitate Blind Denoising in The Visual Hierarchy]

This repository contains data and code from the paper [Retinal Gap Junctions Convert Noise Distribution and Facilitate Blind Denoising in The Visual Hierarchy] by Yang Yue, Kehuan Lun, Liuyuan He, Gan He, Shenjian Zhang, Lei Ma, Jian K Liu, Yonghong Tian, Kai Du, and Tiejun Huang. 

## Installation Guide

### NEURON
The detailed photoreceptor network and gap junction filter run on NEURON simulator. The operating system is Red Hat 4.8.5-36. A quick installation could just be done following the offical instruction (https://neuron.yale.edu/neuron/docs). Note that the NEURON version should be 7.4. 

### Anaconda
Anaconda is is an open source Python distribution. Install Anaconda and Setting up a TensorFlow & Keras environment. The documentation can be found here: https://docs.anaconda.com/. 

## data
This subdirectory contains all images and statictics used for generating figures in our experiments.

## CNNs
This subdirectory contains code for blind denoising and classification in our experiments.

The training and testing dataset for blind denoising come from the paper [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://arxiv.org/abs/1608.03981). The runing sample follows:

`python DnCNN.py --dataset_traiin=train_sample --dataset_test=test_sample --weights=trained_weights.h5 --region` (Training Model)

The training and testing dataset for classification come 
from the paper [Generalisation in humans and deep neural networks](https://arxiv.org/abs/1808.08750). The runing sample follows: 

`python main_resnet50.py --gpus=1 --train_data=p20_train_G-filter --test_data=w0.05_G-filter --save_weights=p20_gap5` (Training Model)

## detailed photoreceptor network
This subdirectory contains code for detailed photoreceptor model, its network and STA analysis. 
Users first run `./nrnivmodl` for `mod` compiling. Then, run `sta_generate_voltage.hoc` to generate voltage curves for STA analysis.

## gap junction filter
This subdirectory contains code for simplified photoreceptor network, i.e. G-filter.
Users first run `./nrnivmodl` for `mod` compiling. Then, run `*_process_sample.hoc` to generate voltage for peak voltage values. Finally, run `*_batch_process_seprate_output.m` to reconstruct filtered image.

## figures
Experiment results in our paper are generated by the following files:

`draw_fig1.m`: Draw the results of STA analysis on the detailed photoreceptor network and the fitting Gaussian curves for 7 samples of time-varying receptive fields.

`draw_fig2_No34_noise_hist.m`: Draw the noise histogram for image No.34 in BSD68.

`draw_fig2_noise_average_distribution.m`: Draw the average noise distributions for all images in BSD68.

`draw_fig3_diff_optimal_distribution.m`: Draw the average noise distributions for all images in BSD68.

`draw_fig3_sensitivity.m`: Draw the average noise distributions for all images in BSD68.

`draw_fig4_fig5.m`: (1) Draw the optimal blind denoising performance curves in fig4, where the testing dataset contains mixed noise and Salt-and-Pepper noise. (2) Draw the blind denoising performance curves under different parameters for parameter sensitivity evaluation in fig5, where the testing dataset contains mixed noise and Salt-and-Pepper noise.

`draw_fig6_optimal_Gaussian_vs_G.m`: Draw the classification accuracies under blind noise attacks, including origin CNNs, Gaussianf-CNNs, and Gf-CNNs.

`draw_fig6_G_filter_with_varting_g.m`: Draw the classification accuracies of Gf-CNNs with different gap junction condutance under blind noise attacks.

