clc;
clear;
load('data/denoising/G-Filtered.mat');
subplot(2,3,1);
plot(-255:255,avg_bins_b,'linewidth',2,'color','b');
hold on;
plot(-255:255,avg_bins_y,'linewidth',2,'color','r');
hold on;
plot(-255:255,avg_bins_q,'linewidth',2,'color','y');
hold on;
plot(-255:255,avg_bins_n,'linewidth',2,'color','k');
xlabel('Difference of Pixel Value','FontSize',12);
ylabel('Probability','FontSize',12);
box off;

load('data/denoising/Gaussian-Filtered-kernel11-sigma5.mat');
subplot(2,3,2);
plot(-255:255,avg_bins_b,'linewidth',2,'color','b');
hold on;
plot(-255:255,avg_bins_y,'linewidth',2,'color','r');
hold on;
plot(-255:255,avg_bins_q,'linewidth',2,'color','y');
hold on;
plot(-255:255,avg_bins_n,'linewidth',2,'color','k');
xlabel('Difference of Pixel Value','FontSize',12);
ylabel('Probability','FontSize',12);
box off;

load('data/denoising/Average-Filtered-kernel11.mat');
subplot(2,3,3);
plot(-255:255,avg_bins_b,'linewidth',2,'color','b');
hold on;
plot(-255:255,avg_bins_y,'linewidth',2,'color','r');
hold on;
plot(-255:255,avg_bins_q,'linewidth',2,'color','y');
hold on;
plot(-255:255,avg_bins_n,'linewidth',2,'color','k');
xlabel('Difference of Pixel Value','FontSize',12);
ylabel('Probability','FontSize',12);
box off;

load('data/denoising/Max-Filtered-kernel11.mat');
subplot(2,3,4);
plot(-255:255,avg_bins_b,'linewidth',2,'color','b');
hold on;
plot(-255:255,avg_bins_y,'linewidth',2,'color','r');
hold on;
plot(-255:255,avg_bins_q,'linewidth',2,'color','y');
hold on;
plot(-255:255,avg_bins_n,'linewidth',2,'color','k');
xlabel('Difference of Pixel Value','FontSize',12);
ylabel('Probability','FontSize',12);
box off;

load('data/denoising/Median-Filtered-kernel11.mat');
subplot(2,3,5);
plot(-255:255,avg_bins_b,'linewidth',2,'color','b');
hold on;
plot(-255:255,avg_bins_y,'linewidth',2,'color','r');
hold on;
plot(-255:255,avg_bins_q,'linewidth',2,'color','y');
hold on;
plot(-255:255,avg_bins_n,'linewidth',2,'color','k');
xlabel('Difference of Pixel Value','FontSize',12);
ylabel('Probability','FontSize',12);
box off;

load('data/denoising/Min-Filtered-kernel11.mat');
subplot(2,3,6);
plot(-255:255,avg_bins_b,'linewidth',2,'color','b');
hold on;
plot(-255:255,avg_bins_y,'linewidth',2,'color','r');
hold on;
plot(-255:255,avg_bins_q,'linewidth',2,'color','y');
hold on;
plot(-255:255,avg_bins_n,'linewidth',2,'color','k');
xlabel('Difference of Pixel Value','FontSize',12);
ylabel('Probability','FontSize',12);
box off;