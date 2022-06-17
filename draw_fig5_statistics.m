clc;
clear;

load('data/Integrated Gradients/sparsity_baseline.mat');
load('data/Integrated Gradients/sparsity_gap1.mat');
load('data/Integrated Gradients/sparsity_gap5.mat');
figure(1);
a=sparsity_baseline(:);
b=sparsity_gap1(:);
c=sparsity_gap5(:);
histogram(a,'Edgecolor','None','Normalization','probability');
hold on;
histogram(b,'Edgecolor','None','Normalization','probability');
hold on;
histogram(c,'Edgecolor','None','Normalization','probability');
hold on;
box off;
xlim([0.4 1]);

figure(2);
boxplot([sparsity_baseline(:),sparsity_gap1(:),sparsity_gap5(:)],'symbol','','Labels',{'g = 0nS','g = 1nS','g = 5 nS'});
ylabel('Sparseness');
title('Statistics for Sparseness')
box off;
ylim([0.3 1]);

