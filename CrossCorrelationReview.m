clear all
clc
close all
root=['/home/sai/AnacondaProjects/Project/MatlabCV/ReviewData'];
acc=zeros(13,1);

for k=1:13
    
file_x=strjoin({'Xset',num2str(k),'10db','mat'},{'_','_','.'});
file_y=strjoin({'Yset',num2str(k),'10db','mat'},{'_','_','.'});

load(fullfile(root,file_x));
% data_x=BaseSig;
data_x=Xset;
load(fullfile(root,file_y));
data_y=Yset;
lagDiff=zeros(899);

for i=2:900
    [r,lags] = xcorr(data_x(i,:),data_x(1,:));
    [~,I] = max(abs(r));
    lagDiff(i-1) = lags(I);
    [~,Idx] = max(abs(r));
        [pks,pklocs] = findpeaks(r,'MinPeakDistance',5);
        [~,Idx] = max(pks);
        MaxIdx(i-1) = pklocs(Idx)-998;
    % To prevent peak correlation occurs at next peak
end 

MaxIdx=MaxIdx';
[~,col]=find(Yset);
col=col(2:end);
chk=find(col==MaxIdx);
acc(k)=100*length(chk)/899;

end

total_acc=mean(acc);

% for -10dB total_acc= 60.699
% for -8dB total_acc= 80.7650
% for -6dB total_acc= 88.8594
% for -4dB total_acc= 93.6767
% for -2dB total_acc= 97.5015
% for 0dB total_acc= 99.1187
% for 2dB total_acc= 99.9401
% for 4dB total_acc= 100
% for 6dB total_acc= 100
% for 8dB total_acc= 100
% for 10dB total_acc= 100