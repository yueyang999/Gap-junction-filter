% visualize attribution map computed by DnCNN_mix_Adam_Integrated_Gradient_multi.py

clc;clear;
load("attribution_map_sample.mat");

x=110;
y=260;
a=ige_data(x-7:x+10,y-7:y+10);
imagesc(a);

% 
%---------------------------blue red colormap
n=[1 42 127 211 252];        
J = zeros(252,3);
%-----------------------------------------------R
J(n(2):n(3),1) =linspace(0,1,n(3)-n(2)+1);
J(n(3):n(4),1) =1;
J(n(4):n(5),1) =linspace(1,0.5,n(5)-n(4)+1);
%-----------------------------------------------G
J(n(2):n(3),2) =linspace(0,1,n(3)-n(2)+1);
J(n(3):n(4),2) =linspace(1,0,n(4)-n(3)+1);
%-----------------------------------------------B
J(n(1):n(2),3) =linspace(0.5,1,n(2)-n(1)+1);
J(n(2):n(3),3) =1;
J(n(3):n(4),3) =linspace(1,0,n(4)-n(3)+1);
%----------------------------------------------------------
rz=get(gca,'clim'); 
rz=max(abs(rz))/3;    
set(gca,'clim',[-rz rz]); 
colormap(J);
box off;
axis off;