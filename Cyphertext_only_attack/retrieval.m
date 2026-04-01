clear all;
%GS algorithm for phase retrieval
M = importdata('./DRPE/decryption4.csv');

phase = importdata('./DRPE/phaseoffsetX.csv');
Amplitude0 = importdata('./W/amplitude.csv');
% image = fft2(fftshift(Amplitude0.*exp(-1i*phase)));  
% a = fftshift((abs(image)).^2)/256/256;
% imshow(a,[]);
itera=2000;
Amplitude=rand(1)*ones(8,8);
scaleFactor=32;
Amplitude=imresize(Amplitude,scaleFactor, 'nearest');
step_size = 0.1;
g0_Fie = Amplitude.*exp(-1i*phase);  
blockSize=[32,32];

for i=1:itera
   G0_Fie = fft2(fftshift(g0_Fie));            %傅立叶变换到频域
   G0_FieNew = sqrt(ifftshift(256*256*M)).*exp(1i*angle(G0_Fie));
   g0_FieNew = ifftshift(ifft2(G0_FieNew));
   A = abs(g0_FieNew);
   A(A>1)=1;
   A(A<0)=0;
   A = blockproc(A, blockSize, @(x) mean(x.data(:)));
   Amplitude = blockproc(Amplitude, blockSize, @(x) mean(x.data(:)));
   a_er = Amplitude - A;    %计算误差矩阵，确定收敛方向
   Amplitude = Amplitude-step_size*a_er;
   Amplitude=imresize(Amplitude,scaleFactor, 'nearest');
   RMS_Fie=sqrt(mean2((a_er.^2)))        %计算均方根误差
   g0_Fie=Amplitude.*exp(-1i*phase); %引入反馈调节
 

end
a1=fftshift(abs(G0_Fie).^2)/256/256;
imshow(a1,[]);

