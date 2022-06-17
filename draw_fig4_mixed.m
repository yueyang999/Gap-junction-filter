clc;clear;

mixed_14_post=load("data\denoising\statictics\Mixed\14PSNR\0nS\postPSNR.txt");
mixed_14_post_1=load("data\denoising\statictics\Mixed\14PSNR\1nS\postPSNR.txt");
mixed_14_post_5=load("data\denoising\statictics\Mixed\14PSNR\5nS\postPSNR.txt");

mixed_16_post=load("data\denoising\statictics\Mixed\16PSNR\0nS\postPSNR.txt");
mixed_16_post_1=load("data\denoising\statictics\Mixed\16PSNR\1nS\postPSNR.txt");
mixed_16_post_5=load("data\denoising\statictics\Mixed\16PSNR\5nS\postPSNR.txt");

mixed_18_post=load("data\denoising\statictics\Mixed\18PSNR\0nS\postPSNR.txt");
mixed_18_post_1=load("data\denoising\statictics\Mixed\18PSNR\1nS\postPSNR.txt");
mixed_18_post_5=load("data\denoising\statictics\Mixed\18PSNR\5nS\postPSNR.txt");

mixed_20_post=load("data\denoising\statictics\Mixed\20PSNR\0nS\postPSNR.txt");
mixed_20_post_1=load("data\denoising\statictics\Mixed\20PSNR\1nS\postPSNR.txt");
mixed_20_post_5=load("data\denoising\statictics\Mixed\20PSNR\5nS\postPSNR.txt");

figure(1);
x=[14 16 18 20];
y=[mean(mixed_14_post) mean(mixed_16_post) mean(mixed_18_post) mean(mixed_20_post)];
y_1=[mean(mixed_14_post_1) mean(mixed_16_post_1) mean(mixed_18_post_1) mean(mixed_20_post_1)];
y_5=[mean(mixed_14_post_5) mean(mixed_16_post_5) mean(mixed_18_post_5) mean(mixed_20_post_5)];

e=[std(mixed_14_post) std(mixed_16_post) std(mixed_18_post) std(mixed_20_post)];
e_1=[std(mixed_14_post_1) std(mixed_16_post_1) std(mixed_18_post_1) std(mixed_20_post_1)];
e_5=[std(mixed_14_post_5) std(mixed_16_post_5) std(mixed_18_post_5) std(mixed_20_post_5)];
subplot(1,2,1);
errorbar(x,y,e,'k');
hold on;
errorbar(x,y_1,e_1,'b');
hold on;
errorbar(x,y_5,e_5,'r');
legend('DnCNN','Hybrid DnCNN(g = 1nS)','Hybrid DnCNN(g = 5nS)');
box off;



mixed_14_post=load("data\denoising\statictics\Mixed\14PSNR\0nS\postSSIM.txt");
mixed_14_post_1=load("data\denoising\statictics\Mixed\14PSNR\1nS\postSSIM.txt");
mixed_14_post_5=load("data\denoising\statictics\Mixed\14PSNR\5nS\postSSIM.txt");

mixed_16_post=load("data\denoising\statictics\Mixed\16PSNR\0nS\postSSIM.txt");
mixed_16_post_1=load("data\denoising\statictics\Mixed\16PSNR\1nS\postSSIM.txt");
mixed_16_post_5=load("data\denoising\statictics\Mixed\16PSNR\5nS\postSSIM.txt");

mixed_18_post=load("data\denoising\statictics\Mixed\18PSNR\0nS\postSSIM.txt");
mixed_18_post_1=load("data\denoising\statictics\Mixed\18PSNR\1nS\postSSIM.txt");
mixed_18_post_5=load("data\denoising\statictics\Mixed\18PSNR\5nS\postSSIM.txt");

mixed_20_post=load("data\denoising\statictics\Mixed\20PSNR\0nS\postSSIM.txt");
mixed_20_post_1=load("data\denoising\statictics\Mixed\20PSNR\1nS\postSSIM.txt");
mixed_20_post_5=load("data\denoising\statictics\Mixed\20PSNR\5nS\postSSIM.txt");


x=[14 16 18 20];
y=[mean(mixed_14_post) mean(mixed_16_post) mean(mixed_18_post) mean(mixed_20_post)];
y_1=[mean(mixed_14_post_1) mean(mixed_16_post_1) mean(mixed_18_post_1) mean(mixed_20_post_1)];
y_5=[mean(mixed_14_post_5) mean(mixed_16_post_5) mean(mixed_18_post_5) mean(mixed_20_post_5)];

e=[std(mixed_14_post) std(mixed_16_post) std(mixed_18_post) std(mixed_20_post)];
e_1=[std(mixed_14_post_1) std(mixed_16_post_1) std(mixed_18_post_1) std(mixed_20_post_1)];
e_5=[std(mixed_14_post_5) std(mixed_16_post_5) std(mixed_18_post_5) std(mixed_20_post_5)];


subplot(1,2,2);
errorbar(x,y,e,'k');
hold on;
errorbar(x,y_1,e_1,'b');
hold on;
errorbar(x,y_5,e_5,'r');
legend('DnCNN','Hybrid DnCNN(g = 1nS)','Hybrid DnCNN(g = 5nS)');
box off;
