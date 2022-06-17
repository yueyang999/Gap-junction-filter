clc;clear;
filterlist={'Original' 'A.Median' 'Average' 'Gaussian' 'Max' 'Median' 'Min' 'G-filter'};
Gaussian_psnr=[14.911	17.683	21.565	22.207	10.209	20.917	10.906	21.780];
Gaussian_ssim=[0.1950	0.2437	0.3703	0.4360	0.2388	0.3544	0.1308	0.5695];

SP_psnr=[15.088	32.288	21.659	22.263	8.489	28.262	10.489	22.275];
SP_ssim=[0.2518	0.9374	0.3930	0.4624	0.1196	0.8313	0.1233	0.5638];

Uniform_psnr=[14.392	16.260	21.158	21.774	10.534	19.091	11.015	21.688];
Uniform_ssim=[0.1778	0.1859	0.3507	0.4138	0.3254	0.2776	0.1719	0.5601];

blind_psnr=[14.1590 19.0992 20.9855 21.5724 8.5531 22.5579 9.9460  23.4669];
blind_ssim=[0.1341	0.2387	0.2790	0.3324	0.1422	0.3300	0.0633	0.4640];


figure(1);
subplot(4,2,1);
[a b]=sort(Gaussian_psnr);
l=filterlist(:,b);
X = categorical(l);
X = reordercats(X,l);
Y = a;
h = bar(X,Y);
colors = get(h, 'FaceColor');
set(h,'edgecolor','none');
ylabel('PSNR');
box off;

subplot(4,2,2);
[a b]=sort(Gaussian_ssim);
l=filterlist(:,b);
X = categorical(l);
X = reordercats(X,l);
Y = a;
h = bar(X,Y);
colors = get(h, 'FaceColor');
set(h,'edgecolor','none');
ylabel('SSIM');
box off;

subplot(4,2,3);
[a b]=sort(SP_psnr);
l=filterlist(:,b);
X = categorical(l);
X = reordercats(X,l);
Y = a;
h = bar(X,Y);
colors = get(h, 'FaceColor');
set(h,'edgecolor','none');
ylabel('PSNR');
box off;
subplot(4,2,4);
[a b]=sort(SP_ssim);
l=filterlist(:,b);
X = categorical(l);
X = reordercats(X,l);
Y = a;
h = bar(X,Y);
colors = get(h, 'FaceColor');
set(h,'edgecolor','none');
ylabel('SSIM');
box off;

subplot(4,2,5);
[a b]=sort(Uniform_psnr);
l=filterlist(:,b);
X = categorical(l);
X = reordercats(X,l);
Y = a;
h = bar(X,Y);
colors = get(h, 'FaceColor');
set(h,'edgecolor','none');
ylabel('PSNR');
box off;
subplot(4,2,6);
[a b]=sort(Uniform_ssim);
l=filterlist(:,b);
X = categorical(l);
X = reordercats(X,l);
Y = a;
h = bar(X,Y);
colors = get(h, 'FaceColor');
set(h,'edgecolor','none');
ylabel('SSIM');
box off;

subplot(4,2,7);
[a b]=sort(blind_psnr);
l=filterlist(:,b);
X = categorical(l);
X = reordercats(X,l);
Y = a;
h = bar(X,Y);
colors = get(h, 'FaceColor');
set(h,'edgecolor','none');
ylabel('PSNR');
box off;
subplot(4,2,8);
[a b]=sort(blind_ssim);
l=filterlist(:,b);
X = categorical(l);
X = reordercats(X,l);
Y = a;
h = bar(X,Y);
colors = get(h, 'FaceColor');
set(h,'edgecolor','none');
ylabel('SSIM');
box off;

