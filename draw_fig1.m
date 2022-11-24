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
plot(-2000:0,sta_curve);
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
figure(2);
for i=1701:50:2001
    count=count+1;
    c=[0.8 0.8 0.8]-(i-1701)*[0.8 0.8 0.8]/400;
%     if(i==1951)
%         plot3((2001-i)/2*ones(15,1),index_flip,w_2d(i,:),'linewidth',2,'color','r');
%     else
        plot3((2001-i)/2*ones(15,1),index_flip,w_2d(i,:),'linewidth',2,'color',c);
%     end
    hold on;
end
xlabel('time before spike (ms)','rotation',15);
ylabel('the distance to center','rotation',-20);
zlabel('weight');
title('The spatial filters with time elapsing');
box off;
% saveas(Fig,'STA','pdf');

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

for i=1701:50:2001
Mytype=fittype('A*exp(-(x)^2/(2*d^2))');%需要拟合的函数类型
X=[-fliplr(index') index'];
X(15)=[];
Y=[fliplr(w_2d(i,:)) w_2d(i,:)];
Y(15)=[];
[cf ,gof]=fit(X',Y',Mytype)%fit函数
xi=-6:0.1:6;
yi=cf.A*exp(-(xi).^2/(2*cf.d^2));%xi，yi为拟合后的函数
plot3((2001-i)/2*ones(121,1),xi,yi,'b');%拟合后的曲线为红色的线
hold on;
% plot3((2001-i)/2*ones(29,1),X,Y,'b');%拟合前的标准曲线为蓝色的线
% hold on;
end

figure(3);
spatio=all_sta(:,1951);
spatio=reshape(spatio,10,10);
imagesc(spatio);
box off;
colorbar;

figure(4);
%gap=1nS
x=-300:50:0;
A=[-0.007035 -0.1959 -0.3518 -0.4476 -0.4899 -0.4903 -0.4477];
sig=[0.1498 0.3993 0.3934 0.3841 0.3829 0.3935 0.4136];
% subplot(1,3,1);
set(gca,'xdir','reverse');
yyaxis left;
plot(x,A);
ylabel('Amplitude');
yyaxis right;
plot(x,sig);
ylabel('Sigma');
box off;
xlabel('time before spike (ms)');

% %gap=5nS
% A=[-0.02368 -0.1738 -0.3043 -0.398 -0.449 -0.4586 -0.4215];
% sig=[0.524 0.5162 0.5134 0.5095 0.5129 0.5248 0.5429];
% subplot(1,3,2);
% set(gca,'xdir','reverse');
% yyaxis left;
% plot(x,A);
% ylabel('Amplitude');
% yyaxis right;
% plot(x,sig);
% ylabel('Sigma');
% box off;
% xlabel('time before spike (ms)');
% 
% %gap=10nS
% A=[-0.02362 -0.1524 -0.2641 -0.3509 -0.4007 -0.4142 -0.3832];
% sig=[0.535 0.5835 0.5915 0.5925 0.6012 0.6099 0.627];
% subplot(1,3,3);
% set(gca,'xdir','reverse');
% yyaxis left;
% plot(x,A);
% ylabel('Amplitude');
% yyaxis right;
% plot(x,sig);
% ylabel('Sigma');
% box off;
% xlabel('time before spike (ms)');