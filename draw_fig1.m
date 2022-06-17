clc;clear;
hz=200;
dt=0.5;
rept=hz/dt;
NOISE_N = 1000000;

% calculate STA
% s2=sprintf('200_0_10_1000000_res_0.001_44.txt');
% resp=load(s2);
% [a b]=findpeaks(-resp);
% parfor j=1:100
%     s1=sprintf('stim/white_stim%d',j-1);
%
%     s3=sprintf('sta/200_0_10_1000000_sta_0.001_%d.txt',j-1);
%     stim=load(s1);
%     stim=stim(1:NOISE_N)*10000;
%     stim=stim';
%     stim=repmat(stim,[rept 1]);
%     stim=stim(:);
%
%
%     sta=zeros(2001,1);
%     for i=1:length(b)
%         if(b(i)<=2000)
%             continue;
%         end
%         sta=sta+stim(b(i)-2000:b(i));
%     end
%     x=0:0.5:1000;
%     sta=sta./length(b)-0.5;
%     dlmwrite(s3, sta);
% end

figure(1);
j=45;
s3=sprintf('data/sta/200_0_10_1000000_sta_0.001_%d.txt',j-1);
sta_curve=load(s3);
plot(sta_curve);
xlabel('time before spike (ms)');
ylabel('weight');
box off;

all_sta=zeros(100,2001);
for j=1:100
    s3=sprintf('data/sta/200_0_10_1000000_sta_0.001_%d.txt',j-1);
    sta_curve=load(s3);
    all_sta(j,:)=sta_curve;
end

for i=1:9
    for j=1:9
        dist(i,j)=sqrt((i-5)^2+(j-5)^2);
    end
end
index=unique(dist(:));

w_2d=zeros(2001,length(index));
for j=1:2001
    w=reshape(all_sta(:,j),10,10);
    for i=1:length(index)
        [x y]=find(dist==index(i));
        temp=0;
        for k=1:length(x)
            temp=temp+w(x(k),y(k));
        end
        w_2d(j,i)=temp/length(x);
    end
end

w_2d_flip=[fliplr(w_2d) w_2d];
index_flip=[-fliplr(index) index];
count=0;
Fig=figure(2);
for i=1701:50:2001
    count=count+1;
    c=[0.8 0.8 0.8]-(i-1701)*[0.8 0.8 0.8]/400;
    if(i==1951)
        plot3((2001-i)/2*ones(15,1),index_flip,w_2d(i,:),'linewidth',2,'color','r');
    else
        plot3((2001-i)/2*ones(15,1),index_flip,w_2d(i,:),'linewidth',2,'color',c);
    end
    hold on;
end
xlabel('time before spike (ms)','rotation',15);
ylabel('the distance to center','rotation',-20);
zlabel('weight');
title('The spatial filters with time elapsing');
box off;
% saveas(Fig,'STA','pdf');

figure(3);
spatio=all_sta(:,1951);
spatio=reshape(spatio,10,10);
imagesc(spatio);
box off;
colorbar;