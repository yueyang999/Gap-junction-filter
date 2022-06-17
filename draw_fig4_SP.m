% clc;clear;
SP_14_post=load("data\denoising\statictics\Salt_and_Pepper\14PSNR\0nS\postPSNR.txt");
SP_14_post_1=load("data\denoising\statictics\Salt_and_Pepper\14PSNR\1nS\postPSNR.txt");
SP_14_post_5=load("data\denoising\statictics\Salt_and_Pepper\14PSNR\5nS\postPSNR.txt");

SP_16_post=load("data\denoising\statictics\Salt_and_Pepper\16PSNR\0nS\postPSNR.txt");
SP_16_post_1=load("data\denoising\statictics\Salt_and_Pepper\16PSNR\1nS\postPSNR.txt");
SP_16_post_5=load("data\denoising\statictics\Salt_and_Pepper\16PSNR\5nS\postPSNR.txt");

SP_18_post=load("data\denoising\statictics\Salt_and_Pepper\18PSNR\0nS\postPSNR.txt");
SP_18_post_1=load("data\denoising\statictics\Salt_and_Pepper\18PSNR\1nS\postPSNR.txt");
SP_18_post_5=load("data\denoising\statictics\Salt_and_Pepper\18PSNR\5nS\postPSNR.txt");

SP_20_post=load("data\denoising\statictics\Salt_and_Pepper\20PSNR\0nS\postPSNR.txt");
SP_20_post_1=load("data\denoising\statictics\Salt_and_Pepper\20PSNR\1nS\postPSNR.txt");
SP_20_post_5=load("data\denoising\statictics\Salt_and_Pepper\20PSNR\5nS\postPSNR.txt");


% figure(1);
x=[14 16 18 20];
y=[mean(SP_14_post) mean(SP_16_post) mean(SP_18_post) mean(SP_20_post)];
y_1=[mean(SP_14_post_1) mean(SP_16_post_1) mean(SP_18_post_1) mean(SP_20_post_1)];
y_5=[mean(SP_14_post_5) mean(SP_16_post_5) mean(SP_18_post_5) mean(SP_20_post_5)];

e=[std(SP_14_post) std(SP_16_post) std(SP_18_post) std(SP_20_post)];
e_1=[std(SP_14_post_1) std(SP_16_post_1) std(SP_18_post_1) std(SP_20_post_1)];
e_5=[std(SP_14_post_5) std(SP_16_post_5) std(SP_18_post_5) std(SP_20_post_5)];
subplot(1,2,1);
errorbar(x,y,e,'k');
hold on;
errorbar(x,y_1,e_1,'b');
hold on;
errorbar(x,y_5,e_5,'r');
legend('DnCNN','Hybrid DnCNN(g = 1nS)','Hybrid DnCNN(g = 5nS)');
box off;

SP_14_post=load("data\denoising\statictics\Salt_and_Pepper\14PSNR\0nS\postSSIM.txt");
SP_14_post_1=load("data\denoising\statictics\Salt_and_Pepper\14PSNR\1nS\postSSIM.txt");
SP_14_post_5=load("data\denoising\statictics\Salt_and_Pepper\14PSNR\5nS\postSSIM.txt");

SP_16_post=load("data\denoising\statictics\Salt_and_Pepper\16PSNR\0nS\postSSIM.txt");
SP_16_post_1=load("data\denoising\statictics\Salt_and_Pepper\16PSNR\1nS\postSSIM.txt");
SP_16_post_5=load("data\denoising\statictics\Salt_and_Pepper\16PSNR\5nS\postSSIM.txt");

SP_18_post=load("data\denoising\statictics\Salt_and_Pepper\18PSNR\0nS\postSSIM.txt");
SP_18_post_1=load("data\denoising\statictics\Salt_and_Pepper\18PSNR\1nS\postSSIM.txt");
SP_18_post_5=load("data\denoising\statictics\Salt_and_Pepper\18PSNR\5nS\postSSIM.txt");

SP_20_post=load("data\denoising\statictics\Salt_and_Pepper\20PSNR\0nS\postSSIM.txt");
SP_20_post_1=load("data\denoising\statictics\Salt_and_Pepper\20PSNR\1nS\postSSIM.txt");
SP_20_post_5=load("data\denoising\statictics\Salt_and_Pepper\20PSNR\5nS\postSSIM.txt");


x=[14 16 18 20];
y=[mean(SP_14_post) mean(SP_16_post) mean(SP_18_post) mean(SP_20_post)];
y_1=[mean(SP_14_post_1) mean(SP_16_post_1) mean(SP_18_post_1) mean(SP_20_post_1)];
y_5=[mean(SP_14_post_5) mean(SP_16_post_5) mean(SP_18_post_5) mean(SP_20_post_5)];

e=[std(SP_14_post) std(SP_16_post) std(SP_18_post) std(SP_20_post)];
e_1=[std(SP_14_post_1) std(SP_16_post_1) std(SP_18_post_1) std(SP_20_post_1)];
e_5=[std(SP_14_post_5) std(SP_16_post_5) std(SP_18_post_5) std(SP_20_post_5)];
subplot(1,2,2);
errorbar(x,y,e,'k');
hold on;
errorbar(x,y_1,e_1,'b'); 
hold on;
errorbar(x,y_5,e_5,'r');
legend('DnCNN','Hybrid DnCNN(g = 1nS)','Hybrid DnCNN(g = 5nS)');
box off;

