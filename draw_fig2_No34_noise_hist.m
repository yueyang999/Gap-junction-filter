clc;
clear;
s1=sprintf("data/denoising/Gray/test034.png");
s2=sprintf("data/denoising/Noisy/Uniform/test034_Uniform.png");
s3=sprintf("data/denoising/Noisy/Salt_and_Pepper/test034_18psnr.png");
s4=sprintf("data/denoising/Noisy/Gaussian/test034_18psnr.png");
s5=sprintf("data/denoising/Noisy/Mixed/test034_combination.png");


s6=sprintf("data/denoising/G-Filtered/Uniform/test034_onepass.png");
s7=sprintf("data/denoising/G-Filtered/Salt_and_Pepper/test034_onepass.png");
s8=sprintf("data/denoising/G-Filtered/Gaussian/test034_onepass.png");
s9=sprintf("data/denoising/G-Filtered/Mixed/test034_onepass.png");

o=imread(s1);
o=o';

a=imread(s2);
a=a';
b=imread(s3);
c=imread(s4);
d=imread(s5);

e=imread(s6);
f=imread(s7);
g=imread(s8);
h=imread(s9);

noisy_U=double(o)-double(a);
noisy_SP=double(o)-double(b);
noisy_G=double(o)-double(c);
noisy_M=double(o)-double(d);

filtered_U=double(o)-double(e);
filtered_SP=double(o)-double(f);
filtered_G=double(o)-double(g);
filtered_M=double(o)-double(h);

category1=char('Before','After');
category2=char('Unifrom noise after G-filter','S&P noise after G-filter','Gaussian noise after G-filter','Mixed noise after G-filter');

figure(1);
subplot(2,2,1);
histogram(noisy_U,'Normalization','probability','EdgeColor','None');
hold on;
histogram(filtered_U,'Normalization','probability','EdgeColor','None');
xlabel('Difference of Pixel Value','FontSize',12);
ylabel('Probability','FontSize',12);
legend(category1(1,:),category1(2,:));
legend('FontName','Times New Roman','FontSize',14);
box off;
subplot(2,2,2);
histogram(noisy_SP,'Normalization','probability','EdgeColor','None');
hold on;
histogram(filtered_SP,'Normalization','probability','EdgeColor','None');
xlabel('Difference of Pixel Value','FontSize',12);
ylabel('Probability','FontSize',12);
legend(category1(1,:),category1(2,:));
legend('FontName','Times New Roman','FontSize',14);
box off;
subplot(2,2,3);
histogram(noisy_G,'Normalization','probability','EdgeColor','None');
hold on;
histogram(filtered_G,'Normalization','probability','EdgeColor','None');
xlabel('Difference of Pixel Value','FontSize',12);
ylabel('Probability','FontSize',12);
legend(category1(1,:),category1(2,:));
legend('FontName','Times New Roman','FontSize',14);
box off;
subplot(2,2,4);
histogram(noisy_M,'Normalization','probability','EdgeColor','None');
hold on;
histogram(filtered_M,'Normalization','probability','EdgeColor','None');
xlabel('Difference of Pixel Value','FontSize',12);
ylabel('Probability','FontSize',12);
legend(category1(1,:),category1(2,:));
legend('FontName','Times New Roman','FontSize',14);
box off;


figure(2);
h=histogram(filtered_U,'Normalization','probability','EdgeColor','None');
xlim([-60 60]);
hold on;
h=histogram(filtered_SP,'Normalization','probability','EdgeColor','None');
xlim([-60 60]);
hold on;
histogram(filtered_G,'Normalization','probability','EdgeColor','None');
xlim([-60 60]);
hold on;
histogram(filtered_M,'Normalization','probability','EdgeColor','None');
xlim([-60 60]);
xlabel('Difference of Pixel Value','FontSize',12);
ylabel('Probability','FontSize',12);
legend(category2(1,:),category2(2,:),category2(3,:),category2(4,:));
legend('FontName','Times New Roman','FontSize',14);
box off;
