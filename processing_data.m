data=scan_line(1:1000);
data_final=zeros(2000,900);
zerovec=zeros(1,1000);

for i=1:900
    data_final(i:i+1000-1,i)=data;
end