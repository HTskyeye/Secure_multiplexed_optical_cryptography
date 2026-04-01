phasetest = importdata('./DRPE/phase7.csv');
ftest=fft2(fftshift(Amplitude.*exp(-1i*phasetest)));
dtest=fftshift((abs(ftest)).^2)/256/256;
imshow(dtest,[]);
imwrite(dtest,'./DRPE/decryption7.jpg');
%imwrite(Amplitude,'./DRPE/key.jpg');
% Amplitude0 = importdata('./WO/amplitude.csv');
% imwrite(Amplitude0,'./WO/key0.jpg');