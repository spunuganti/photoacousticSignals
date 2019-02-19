clear all
clc
close all
root=['/home/sai/AnacondaProjects/Project/MatlabCV/MultiReviewData3Sigs'];
acc=zeros(13,1);

%Loading in the templates
template=zeros(3,101);
for j=1:3
    FileName = strjoin({'reviewdata',num2str(j),'.mat'},{''});
    FolderName = '/home/sai/AnacondaProjects/Project/MatlabCV/ReviewData/';
    Path = [FolderName,FileName];
    load(Path);
    scan_line=downsample(scan_line,10);
    template(j,:)=scan_line;
end

count=1;
    
for snr=10:-2:-10
for k=1:13
    
file_x=strjoin({'Xset',num2str(k),num2str(snr),'db','mat'},{'_','_','','.'});
file_y=strjoin({'Yset',num2str(k),num2str(snr),'db','mat'},{'_','_','','.'});

load(fullfile(root,file_x));
data_x=BaseSig;
%data_x=Xset;
load(fullfile(root,file_y));
data_y=Yset;
lagDiff=zeros(899);
chk=zeros(1,900);
for i=1:900
      [r1,lags1] = xcorr(data_x(i,:),template(3,:));
%       [r2,lags2] = xcorr(data_x(i,:),template(2,:));
%       [r3,lags3] = xcorr(data_x(i,:),template(3,:));
%     [~,I] = max(abs(r));
%     lagDiff(i-1) = lags(I);
%     [~,Idx] = max(abs(r));
      [pks1,pklocs1] = findpeaks(r1,'MinPeakDistance',5);
%       [pks2,pklocs2] = findpeaks(r2,'MinPeakDistance',5);
%       [pks3,pklocs3] = findpeaks(r3,'MinPeakDistance',5);
        [~,Idx1] = max(pks1);
%         [~,Idx2] = max(pks2);
%         [~,Idx3] = max(pks3);
        MaxIdx1 = pklocs1(Idx1)-998;
%         MaxIdx2 = pklocs2(Idx2)-998;
%         MaxIdx3 = pklocs3(Idx3)-998;
    % To prevent peak correlation occurs at next peak
        col=find(Yset(i,:));
        [temp,originalpos] = sort( pks1, 'descend' );
        n = temp(1:3);
        p=originalpos(1:3);
        cor=pklocs1(p)-998;
        chk(i)=isequal(col,sort(cor));
       % MaxIdx=[MaxIdx1;MaxIdx2;MaxIdx3];
end 

% MaxIdx=MaxIdx';
% [~,col]=find(Yset);
% col=col(2:end);
% chk=find(col==MaxIdx);
acc(k)=100*sum(chk)/900;

end

total_acc(count)=mean(acc);
count=count+1;
end

% for -10dB total_acc= 38.4274
% for -8dB total_acc= 57.8034
% for -6dB total_acc= 72.5812
% for -4dB total_acc= 81.9316
% for -2dB total_acc= 84.3675
% for 0dB total_acc= 85.1709
% for 2dB total_acc= 85.5812
% for 4dB total_acc= 84.7436
% for 6dB total_acc= 85.1282
% for 8dB total_acc= 84.6068
% for 10dB total_acc= 85.2393