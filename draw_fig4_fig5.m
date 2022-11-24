clc;
clear;
x=[14 16 18 20];
y_mixed=[15.8851   16.9747   19.4217   21.9800];
y=[17.1361   20.0889   22.1778   23.6440];
y_1=[22.5087   23.2510   23.6564   24.0798];
y_3=[21.9345 22.8292 23.5812 24.1273];
y_5=[21.7116   22.8264   23.6294   24.0692];
y_10=[20.7097 22.4005 23.2192 23.6282];



y_kernel11_0_5=[13.089178 13.8428545 14.410491 14.913602];
y_kernel11_1=[14.827408 16.260525 17.69619 19.196121];
y_kernel11_5=[21.264194 21.693314 21.866274 21.900427];
y_kernel11_10=[20.839653 21.299107 21.539892 21.644283];

y_kernel3_5=[16.201002 16.333542 16.207228 16.324990];
y_kernel5_5=[15.449856 17.284437 19.033464 20.478703];
y_kernel7_5=[19.795969 21.102310 21.803070 22.292082];

y_1_mixed=[21.242012   22.105367   23.486645   23.797329];
y_3_mixed=[21.8288 22.5324 23.6171 23.997];
y_5_mixed=[21.85636   22.667143   23.77315   24.045502]; 
y_10_mixed=[20.4731 21.2638 21.8604 22.1426];

y_kernel11_0_5_mixed=[13.427055 14.409437 15.111613 15.288625];
y_kernel11_1_mixed=[14.0991125 15.538485 17.50756 18.840492];
y_kernel11_5_mixed=[21.357952 21.696712 21.876684 21.90121];
y_kernel11_10_mixed=[21.045849 21.352095 21.568844 21.6450];

y_kernel3_5_mixed=[16.201002 16.333542 16.207228 16.324990];
y_kernel5_5_mixed=[15.412551 16.875813 18.92229 20.161339];
y_kernel7_5_mixed=[20.201384 21.192001 21.915222 22.22548];


figure(1);
subplot(1,2,1);
plot(x,y,'k');
hold on;
plot(x,y_kernel11_5,'b');
hold on;
plot(x,y_1,'r');

box off;
xlabel('Original Image Quality(dB)');
ylabel('Processed Image Quality');
ylim([14 25]);
legend('DnCNN','Gaussianf-DnCNN','Gf-DnCNN');

subplot(1,2,2);
plot(x,y_mixed,'k');
hold on;
plot(x,y_kernel11_5_mixed,'b');
hold on;
plot(x,y_5_mixed,'r');

box off;
xlabel('Original Image Quality(dB)');
ylabel('Processed Image Quality');
ylim([14 25]);
legend('DnCNN','Gaussianf-DnCNN','Gf-DnCNN');



figure(2);
subplot(2,3,1);
plot(x,y_1,color=[1 0 0]);
hold on;
plot(x,y_3,color=[1 0 0]/4*3);
hold on;
plot(x,y_5,color=[1 0 0]/2);
hold on;
plot(x,y_10,color=[1 0 0]/4);
box off;
xlabel('Original Image Quality(dB)');
ylabel('Processed Image Quality');
ylim([14 25]);
legend('g = 1nS','g = 3nS','g = 5nS','g = 10nS');

subplot(2,3,2);
% plot(x,y_kernel11_0_5);
% hold on;
plot(x,y_kernel11_0_5,color=[0 0 1]);
hold on;
plot(x,y_kernel11_1,color=[0 0 1]/4*3);
hold on;
plot(x,y_kernel11_5,color=[0 0 1]/2);
hold on;
plot(x,y_kernel11_10,color=[0 0 1]/4);
box off;
xlabel('Original Image Quality(dB)');
ylabel('Processed Image Quality');
ylim([14 25]);
legend('Gaussian Filter(kernel=11, sigma = 0.5)','Gaussian Filter(kernel=11, sigma = 1)','Gaussian Filter(kernel=11, sigma = 5)','Gaussian Filter(kernel=11, sigma = 10)');

subplot(2,3,3);
plot(x,y_kernel3_5,color=[0 0 1]);
hold on;
plot(x,y_kernel5_5,color=[0 0 1]/4*3);
hold on;
plot(x,y_kernel7_5,color=[0 0 1]/2);
hold on;
plot(x,y_kernel11_5,color=[0 0 1]/4);

box off;
xlabel('Original Image Quality(dB)');
ylabel('Processed Image Quality');
ylim([14 25]);
legend('Gaussian Filter(kernel=3, sigma = 5)','Gaussian Filter(kernel=5, sigma = 5)','Gaussian Filter(kernel=7, sigma = 5)','Gaussian Filter(kernel=11, sigma = 5)');


subplot(2,3,4);
plot(x,y_1_mixed,color=[1 0 0]);
hold on;
plot(x,y_3_mixed,color=[1 0 0]/4*3);
hold on;
plot(x,y_5_mixed,color=[1 0 0]/2);
hold on;
plot(x,y_10_mixed,color=[1 0 0]/4);
box off;
xlabel('Original Image Quality(dB)');
ylabel('Processed Image Quality');
ylim([14 25]);
legend('G-filter(g = 1nS)','G-filter(g = 3nS)','G-filter(g = 5nS)','G-filter(g = 10nS)');

subplot(2,3,5);
% plot(x,y_kernel11_0_5_mixed);
% hold on;
plot(x,y_kernel11_0_5_mixed,color=[0 0 1]);
hold on;
plot(x,y_kernel11_1_mixed,color=[0 0 1]/4*3);
hold on;
plot(x,y_kernel11_5_mixed,color=[0 0 1]/2);
hold on;
plot(x,y_kernel11_10_mixed,color=[0 0 1]/4);
box off;
xlabel('Original Image Quality(dB)');
ylabel('Processed Image Quality');
ylim([14 25]);
legend('Gaussian Filter(kernel=11, sigma = 0.5)','Gaussian Filter(kernel=11, sigma = 1)','Gaussian Filter(kernel=11, sigma = 5)','Gaussian Filter(kernel=11, sigma = 10)');

subplot(2,3,6);
plot(x,y_kernel3_5_mixed,color=[0 0 1]);
hold on;
plot(x,y_kernel5_5_mixed,color=[0 0 1]/4*3);
hold on;
plot(x,y_kernel7_5_mixed,color=[0 0 1]/2);
hold on;
plot(x,y_kernel11_5_mixed,color=[0 0 1]/4);

box off;
xlabel('Original Image Quality(dB)');
ylabel('Processed Image Quality');
ylim([14 25]);
legend('Gaussian Filter(kernel=3, sigma = 5)','Gaussian Filter(kernel=5, sigma = 5)','Gaussian Filter(kernel=7, sigma = 5)','Gaussian Filter(kernel=11, sigma = 5)');

