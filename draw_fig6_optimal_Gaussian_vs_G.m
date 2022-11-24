clc;clear;
humans=[0.805,0.79625,0.78125,0.75125,0.60875,0.45625,0.1675,0.06];

ResNet50=[0.659821428571429,0.529464285714286,0.3625,0.127678571428571,0.0642857142857143,0.0625,0.0598214285714286,0.0616071428571429]; %source

G_resnet50=[0.8240    0.6248    0.5780    0.5269    0.5033    0.4786    0.4181    0.2816];
Gaussian_resnet50=[0.7493 0.5658 0.4326 0.4133 0.3805 0.2323 0.1846 0.1068];



vgg19=[0.879464286,0.792857143,0.710714286,0.49375,0.240178571,0.100892857,0.066964286,0.0625];
G_vgg19=[0.8822    0.7773    0.7151    0.6482    0.5924    0.5506    0.4936    0.2949];
Gaussian_vgg19=[0.8589 0.7493 0.6761 0.6371 0.5441 0.4922 0.4425 0.1043];


inceptionv3=[0.858928571,0.758928571,0.604464286,0.282142857,0.086607143,0.065178571,0.065178571,0.064285714]; %source
G_inceptionv3=[0.8914    0.7649    0.7128    0.5933    0.4646    0.4301    0.3445    0.2846];
Gaussian_inceptionv3=[0.8569  0.5221 0.4517 0.4271 0.3972 0.3684 0.2989 0.1991];

x=[0 0.03 0.05 0.1 0.2 0.35 0.6 0.9];
figure(1);

subplot(1,3,1);

plot(x,G_resnet50,'r','Marker','^','MarkerSize',8,'LineWidth',1);
hold on;
plot(x,Gaussian_resnet50,'b','Marker','x','MarkerSize',8,'LineWidth',1);
hold on;
plot(x,humans,'k','Marker','square','MarkerSize',8,'MarkerFaceColor','None','LineWidth',1);
hold on;
plot(x,ResNet50,'k--','LineWidth',1);
hold on;
xlabel('Uniform noise width');
ylabel('Accuracy');
legend('G-ResNet50','Gaussian-ResNet50','Humans','ResNet50');
box off;

subplot(1,3,2);
plot(x,G_vgg19,'r','Marker','^','MarkerSize',8,'LineWidth',1);
hold on;
plot(x,Gaussian_vgg19,'b','Marker','x','MarkerSize',8,'LineWidth',1);
hold on;
plot(x,humans,'k','Marker','square','MarkerSize',8,'MarkerFaceColor','None','LineWidth',1);
hold on;
plot(x,inceptionv3,'k--','LineWidth',1);
hold on;


xlabel('Uniform noise width');
ylabel('Accuracy');
legend('G-VGG19','Gaussian-VGG19','Humans','VGG19');
box off;


subplot(1,3,3);
plot(x,G_inceptionv3,'r','Marker','^','MarkerSize',8,'LineWidth',1);
hold on;
plot(x,Gaussian_inceptionv3,'b','Marker','x','MarkerSize',8,'LineWidth',1);
hold on;
plot(x,humans,'k','Marker','square','MarkerSize',8,'MarkerFaceColor','None','LineWidth',1);
hold on;
plot(x,vgg19,'k--','LineWidth',1);
hold on;

xlabel('Uniform noise width');
ylabel('Accuracy');
legend('G-Inceptionv3','Gaussian-Inceptionv3','Humans','Inceptionv3');
box off;




