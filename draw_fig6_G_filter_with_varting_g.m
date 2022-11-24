clc;clear;
humans=[0.805,0.79625,0.78125,0.75125,0.60875,0.45625,0.1675,0.06];

ResNet50=[0.659821428571429,0.529464285714286,0.3625,0.127678571428571,0.0642857142857143,0.0625,0.0598214285714286,0.0616071428571429]; %source

G_resnet50_g1=[0.8334 0.6278 0.5081 0.4841 0.4596  0.4696 0.4516 0.2578];
G_resnet50_g3=[0.8244 0.6238 0.5515 0.4958 0.4818  0.4768 0.4493 0.3078];
G_resnet50_g5=[0.8240 0.6248 0.5780 0.5269 0.5033  0.4786 0.4181 0.2816];
vgg19=[0.879464286,0.792857143,0.710714286,0.49375,0.240178571,0.100892857,0.066964286,0.0625];

G_vgg19_g1=[0.8775 0.8123 0.7671 0.6321 0.5848 0.5114 0.4017 0.2422];
G_vgg19_g3=[0.8790 0.7918 0.7909 0.6432 0.5901 0.5349 0.4463 0.2436];
G_vgg19_g5=[0.8822 0.7773 0.7151 0.6482 0.5924 0.5506 0.4936 0.2949];

inceptionv3=[0.858928571,0.758928571,0.604464286,0.282142857,0.086607143,0.065178571,0.065178571,0.064285714];
G_inceptionv3_g1=[0.8912 0.7163 0.7049 0.6530 0.5579 0.4295 0.3671 0.2624];
G_inceptionv3_g3=[0.8889 0.7349 0.7032 0.6019 0.5461 0.4913 0.4048 0.2236];
G_inceptionv3_g5=[0.8914 0.7649 0.7128 0.5933 0.4646 0.4301 0.3445 0.2846];

% resnet50=[G_resnet50_g1;G_resnet50_g3;G_resnet50_g5];
% mean(max(resnet50)-min(resnet50));
% 
% vgg19=[G_vgg19_g1;G_vgg19_g3;G_vgg19_g5];
% mean(max(vgg19)-min(vgg19));
% 
% inceptionv3=[G_inceptionv3_g1;G_inceptionv3_g3;G_inceptionv3_g5];
% mean(max(inceptionv3)-min(inceptionv3));



x=[0 0.03 0.05 0.1 0.2 0.35 0.6 0.9];
figure(1);

subplot(1,3,1);
plot(x,G_resnet50_g1,'Marker','^','MarkerSize',8,'LineWidth',1,color=[1 0 0]/2);
hold on;
plot(x,G_resnet50_g3,'Marker','x','MarkerSize',8,'LineWidth',1,color=[1 0 0]*3/4);
hold on;
plot(x,G_resnet50_g5,'Marker','square','MarkerSize',8,'MarkerFaceColor','None','LineWidth',1,color=[1 0 0]);
hold on;
plot(x,humans,'k','LineWidth',1);
hold on;
plot(x,ResNet50,'k--','LineWidth',1);
hold on;

ylim([0 1]);

xlabel('Uniform noise width');
ylabel('Accuracy');
legend('gap1','gap3','gap5','human','Resnet50');
box off;
title('Resnet50 with G-filter');

subplot(1,3,2);
plot(x,G_vgg19_g1,'Marker','^','MarkerSize',8,'LineWidth',1,color=[1 0 0]/2);
hold on;
plot(x,G_vgg19_g3,'Marker','x','MarkerSize',8,'LineWidth',1,color=[1 0 0]*3/4);
hold on;
plot(x,G_vgg19_g5,'Marker','square','MarkerSize',8,'MarkerFaceColor','None','LineWidth',1,color=[1 0 0]);
hold on;
plot(x,humans,'k','LineWidth',1);
hold on;
plot(x,vgg19,'k--','LineWidth',1);
hold on;
ylim([0 1]);

xlabel('Uniform noise width');
ylabel('Accuracy');
legend('gap1','gap3','gap5','human','VGG19');
box off;
title('VGG19 with G-filter');


subplot(1,3,3);
plot(x,G_inceptionv3_g1,'Marker','^','MarkerSize',8,'LineWidth',1,color=[1 0 0]/2);
hold on;
plot(x,G_inceptionv3_g3,'Marker','x','MarkerSize',8,'LineWidth',1,color=[1 0 0]*3/4);
hold on;
plot(x,G_inceptionv3_g5,'Marker','square','MarkerSize',8,'MarkerFaceColor','None','LineWidth',1,color=[1 0 0]);
hold on;
plot(x,humans,'k','LineWidth',1);
hold on;
plot(x,inceptionv3,'k--','LineWidth',1);
hold on;
ylim([0 1]);

xlabel('Uniform noise width');
ylabel('Accuracy');
legend('gap1','gap3','gap5','human','Inception V3');
box off;
title('Inception V3 with G-filter');




