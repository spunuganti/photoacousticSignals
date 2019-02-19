% clear all

% load('USprobe.mat');
% load('ActiveSource.mat');
fs=4e+7;

data=(Xset(35,:));
d = (fdesign.bandpass('N,Fc1,Fc2',10,3.25e6,3.75e6,4e7));
Hd = design(d,'butter');
cln=filter(Hd,data);
%filtfilt

NFFT = length(data);              % Number of FFT points
F = (0 : 1/NFFT : 1/2-1/NFFT)*fs; % Frequency vector

y = fft(cln,NFFT);
y(1) = 0; % remove the DC component for better visualization

plot(F,abs(y(1:NFFT/2)));

[pks,pklocs] = findpeaks(abs(y(1,1:NFFT/2)),'MinPeakDistance',5);


