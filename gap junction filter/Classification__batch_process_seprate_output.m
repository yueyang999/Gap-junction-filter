
clc;clear;
advlist=char('w0','w0.03', 'w0.05'，'w0.1'，'w0.2', 'w0.35','w0.6', ,'w0.9' );

classlist=char('knife','bicycle','bear','truck','airplane','clock','boat','car','keyboard','oven','cat','bird','elephant','chair','bottle','dog');
gaplist=[3];
sz1=size(advlist);
for x=1:sz1(1)
    newfilename=strip(advlist(x,:));
    for j=1:length(gaplist)
        gap=gaplist(j);
        for t=1:16
            newclassname=strip(classlist(t,:));
            disp(newclassname);
            s3=sprintf('%s_map_gap_%g_PR/%s',newfilename,gap,newclassname);
            if ~exist(s3)
                mkdir(s3);
            end
            s2=sprintf('%s/profile/%s/',newfilename,newclassname);
            filelist=dir([s2,'*.txt']);
            parfor p=1:length(filelist)
                k=floor(p/1000);
                a=[];
                for r=0:23
                    s1=sprintf('%s_map_gap_%g_result/%s/%d/%d_%g_min_voltage_%g_%g.txt',newfilename,gap,newclassname,k,p,gap,gap*0.001,r);
                    a=[a;load(s1)];
                end
                s2=sprintf('%s_map_gap_%g_PR/%s/%d.png',newfilename,gap,newclassname,p-1);
                
                b=sortrows(a,1);
                b=b(:,2);
                b=(b-min(b))./(max(abs(b))-min(abs(b)))*255;
                b=uint8(b);
                c=reshape(b,32,32);
                c=255-uint8(c);
                imwrite(c,s2);
            end
        end
    end
end