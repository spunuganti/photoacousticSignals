clear all
clc
close all
root=['/home/sai/AnacondaProjects/Project/MatlabCV/TrainingData'];
acc=zeros(13,1);

for k=1
    
file_x=strjoin({'Xset',num2str(k),'10db','mat'},{'_','_','.'});
file_y=strjoin({'Yset',num2str(k),'10db','mat'},{'_','_','.'});

load(fullfile(root,file_x));
% data_x=BaseSig;
data_x=Xset;
load(fullfile(root,file_y));
data_y=Yset;


for i=1:900
       indices=find(data_x(i,:)>0);
       [pks,pklocs] = findpeaks(data_x(i,indices),'MinPeakDistance',10);
        [~,Idx] = max(pks);
        MaxIdx(i) = pklocs(Idx);
    % To prevent peak correlation occurs at next peak
end 

MaxIdx=MaxIdx';
[~,col]=find(Yset);
chk=find(col==MaxIdx);
acc(k)=100*length(chk)/899;

end

total_acc=mean(acc);

% for -10db 0.0171
%for -8db 0.0428
% -6db 0.0171
% -4db 0.0086
% -2db 0.0171
% 0db 0.0171
% 2db 0.0171
% 4db 0.0086
% 6db 0.0086
% 8db 0.0171
% 10db 0.0171

