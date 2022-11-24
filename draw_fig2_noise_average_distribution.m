clc;
clear;
a=zeros(68,321,481);
b=zeros(68,321,481);
x=zeros(68,321,481);
y=zeros(68,321,481);
p=zeros(68,321,481);
q=zeros(68,321,481);
m=zeros(68,321,481);
n=zeros(68,321,481);

for i=1:68
    if(i<10)
        s1=sprintf("data/denoising/Gray/test00%d.png",i);
        s2=sprintf("data/denoising/Noisy/Uniform/test00%d_Uniform.png",i);
        s3=sprintf("data/denoising/Noisy/Salt_and_Pepper/test00%d_18psnr.png",i);
        s6=sprintf("data/denoising/Noisy/Gaussian/test00%d_18psnr.png",i);
        s8=sprintf("data/denoising/Noisy/Mixed/test00%d_combination.png",i);

        s4=sprintf("data/denoising/G-Filtered/Uniform/test00%d_onepass.png",i);
        s5=sprintf("data/denoising/G-Filtered/Salt_and_Pepper/test00%d_onepass.png",i);
        s7=sprintf("data/denoising/G-Filtered/Gaussian/test00%d_onepass.png",i);
        s9=sprintf("data/denoising/G-Filtered/Mixed/test00%d_onepass.png",i);
        
        
    else
        s1=sprintf("data/denoising/Gray/test0%d.png",i);
        s2=sprintf("data/denoising/Noisy/Uniform/test0%d_Uniform.png",i);
        s3=sprintf("data/denoising/Noisy/Salt_and_Pepper/test0%d_18psnr.png",i);
        s6=sprintf("data/denoising/Noisy/Gaussian/test0%d_18psnr.png",i);
        s8=sprintf("data/denoising/Noisy/Mixed/test0%d_combination.png",i);

        s4=sprintf("data/denoising/G-Filtered/Uniform/test0%d_onepass.png",i);
        s5=sprintf("data/denoising/G-Filtered/Salt_and_Pepper/test0%d_onepass.png",i);
        s7=sprintf("data/denoising/G-Filtered/Gaussian/test0%d_onepass.png",i);
        s9=sprintf("data/denoising/G-Filtered/Mixed/test0%d_onepass.png",i);
        
        
    end
    origin=double(imread(s1));
    sz=size(origin);
    if(sz(1)>sz(2))
        origin=origin';
    end
    noisy_U=double(imread(s2));
    sz=size(noisy_U);
    if(sz(1)>sz(2))
        noisy_U=noisy_U';
    end
    noisy_SP=double(imread(s3));
    
    noisy_G=double(imread(s6));
    noisy_M=double(imread(s8));
    
    filtered_U=double(imread(s4));
    filtered_SP=double(imread(s5));
    filtered_G=double(imread(s7));
    filtered_M=double(imread(s9));
    
    a(i,:,:)=origin-noisy_U;
    b(i,:,:)=origin-filtered_U;
    x(i,:,:)=origin-noisy_SP;
    y(i,:,:)=origin-filtered_SP;
    
    p(i,:,:)=origin-noisy_G;
    q(i,:,:)=origin-filtered_G;
    m(i,:,:)=origin-noisy_M;
    n(i,:,:)=origin-filtered_M;
    
end

bins_a=zeros(1,511);
bins_b=zeros(1,511);
bins_x=zeros(1,511);
bins_y=zeros(1,511);

bins_p=zeros(1,511);
bins_q=zeros(1,511);
bins_m=zeros(1,511);
bins_n=zeros(1,511);

avg_bins_a=zeros(1,511);
avg_bins_b=zeros(1,511);
avg_bins_x=zeros(1,511);
avg_bins_y=zeros(1,511);

avg_bins_p=zeros(1,511);
avg_bins_q=zeros(1,511);
avg_bins_m=zeros(1,511);
avg_bins_n=zeros(1,511);

for i=1:68
    for j=-255:255
        temp=a(i,:,:);
        temp=temp(:);
        bins_a(j+256)=length(temp(temp==j));
        temp=b(i,:,:);
        temp=temp(:);
        bins_b(j+256)=length(temp(temp==j));
        temp=x(i,:,:);
        temp=temp(:);
        bins_x(j+256)=length(temp(temp==j));
        temp=y(i,:,:);
        temp=temp(:);
        bins_y(j+256)=length(temp(temp==j));
        
        temp=p(i,:,:);
        temp=temp(:);
        bins_p(j+256)=length(temp(temp==j));
        temp=q(i,:,:);
        temp=temp(:);
        bins_q(j+256)=length(temp(temp==j));
        temp=m(i,:,:);
        temp=temp(:);
        bins_m(j+256)=length(temp(temp==j));
        temp=n(i,:,:);
        temp=temp(:);
        bins_n(j+256)=length(temp(temp==j));
    end
    avg_bins_a=avg_bins_a+bins_a/sum(bins_a);
    avg_bins_b=avg_bins_b+bins_b/sum(bins_b);
    avg_bins_x=avg_bins_x+bins_x/sum(bins_x);
    avg_bins_y=avg_bins_y+bins_y/sum(bins_y);
    
    avg_bins_p=avg_bins_p+bins_p/sum(bins_p);
    avg_bins_q=avg_bins_q+bins_q/sum(bins_q);
    avg_bins_m=avg_bins_m+bins_m/sum(bins_m);
    avg_bins_n=avg_bins_n+bins_n/sum(bins_n);
end
avg_bins_a=avg_bins_a/68;
avg_bins_b=avg_bins_b/68;
avg_bins_x=avg_bins_x/68;
avg_bins_y=avg_bins_y/68;

avg_bins_p=avg_bins_p/68;
avg_bins_q=avg_bins_q/68;
avg_bins_m=avg_bins_m/68;
avg_bins_n=avg_bins_n/68;


category1=char('Before PR Filter','After G-filter');
category2=char('Before PR Filter','After G-filter');
category3=char('Before PR Filter','After G-filter');
category4=char('Before PR Filter','After G-filter');
category5=char('Unifrom noise after G-filter','S&P noise after G-filter','Gaussian noise after G-filter','Mixed noise after G-filter');
figure(1);
subplot(1,5,1);
plot(-255:255,avg_bins_a,'linewidth',2,'color','k');
hold on;
plot(-255:255,avg_bins_b,'linewidth',2,'color','r');
xlabel('Difference of Pixel Value','FontSize',12);
ylabel('Probability','FontSize',12);
% legend(category1(1,:),category1(2,:));
% legend('FontName','Times New Roman','FontSize',14);
box off;

subplot(1,5,2);
plot(-255:255,avg_bins_x,'linewidth',2,'color','k');
hold on;
plot(-255:255,avg_bins_y,'linewidth',2,'color','b');
xlabel('Difference of Pixel Value','FontSize',12);
ylabel('Probability','FontSize',12);
% legend(category2(1,:),category2(2,:));
% legend('FontName','Times New Roman','FontSize',14);
ylim([0 0.025]);
box off;

subplot(1,5,3);
plot(-255:255,avg_bins_p,'linewidth',2,'color','k');
hold on;
plot(-255:255,avg_bins_q,'linewidth',2,'color','b');
xlabel('Difference of Pixel Value','FontSize',12);
ylabel('Probability','FontSize',12);
% legend(category3(1,:),category3(2,:));
% legend('FontName','Times New Roman','FontSize',14);
ylim([0 0.025]);
box off;

subplot(1,5,4);
plot(-255:255,avg_bins_m,'linewidth',2,'color','k');
hold on;
plot(-255:255,avg_bins_n,'linewidth',2,'color','b');
xlabel('Difference of Pixel Value','FontSize',12);
ylabel('Probability','FontSize',12);
% legend(category4(1,:),category4(2,:));
% legend('FontName','Times New Roman','FontSize',14);
ylim([0 0.025]);
box off;

subplot(1,5,5);
plot(-255:255,avg_bins_b,'linewidth',2,'color','r');
hold on;
plot(-255:255,avg_bins_y,'linewidth',2,'color','b');
hold on;
plot(-255:255,avg_bins_q,'linewidth',2,'color','c');
hold on;
plot(-255:255,avg_bins_n,'linewidth',2,'color','y');
xlabel('Difference of Pixel Value','FontSize',12);
ylabel('Probability','FontSize',12);
%legend(category5(1,:),category5(2,:),category5(3,:),category5(4,:));
%legend('FontName','Times New Roman','FontSize',14);
box off;