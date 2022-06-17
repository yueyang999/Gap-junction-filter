clc;
clear;
record_x=[305   209   290   218   271   260   248   239];
record_y=[64    43    31    74    50    75    31    54];

r_g0=[ 10     12     11     11    12     12   10    12 ];
figure(1);
a=imread("data/Integrated Gradients/fig5_gap0.jpg");
a=imresize(a,[481 321]);
a=imrotate(a,90);
c=uint8(zeros(321,481,3));
for i=1:length(record_x)
    s1=sprintf('data/Integrated Gradients/Gaussian_baseline_SP_test_%d.png',i);
    b=imread(s1);
    center_x=record_x(i);
    center_y=record_y(i);
    r1=r_g0(i);
    for x=1:length(b)
        for y=1:length(b)
            if(sqrt((x-r1)^2+(y-r1)^2)>r1)
                b(x,y,:)=0;
            end
        end
    end
    for x=center_x-r1:center_x+r1
        for y=center_y-r1:center_y+r1
            if(b(x-(center_x-r1)+1,y-(center_y-r1)+1,:)~=0)
                a(y+1,x+1,:)=0;
                c(y+1,x+1,:)=b(x-(center_x-r1)+1,y-(center_y-r1)+1,:);
            end
        end
    end
end
d=imadd(a,c);
d=d(10:90,190:320,:);
imshow(d);

r_g5=[ 13     13     13     13    12     12   12    12 ];
figure(2);
a=imread("data/Integrated Gradients/fig5_gap5.jpg");
a=imresize(a,[481 321]);
a=imrotate(a,90);
c=uint8(zeros(321,481,3));

for i=1:length(record_x)
    s1=sprintf('data/Integrated Gradients/Gaussian_gap_5_SP_test_%d.png',i);
    b=imread(s1);
    center_x=record_x(i);
    center_y=record_y(i);
    r1=r_g5(i);
    for x=1:length(b)
        for y=1:length(b)
            if(sqrt((x-r1)^2+(y-r1)^2)>r1)
                b(x,y,:)=0;
            end
        end
    end
    for x=center_x-r1:center_x+r1
        for y=center_y-r1:center_y+r1
            if(b(x-(center_x-r1)+1,y-(center_y-r1)+1,:)~=0)
                a(y+1,x+1,:)=0;
                c(y+1,x+1,:)=b(x-(center_x-r1)+1,y-(center_y-r1)+1,:);
            end
        end
    end
end
d=imadd(a,c);
d=d(10:90,190:320,:);
imshow(d);
