% Clean workspace
clear all; close all; clc

%% Loading cat and dog data

load('catData.mat')
load('dogData.mat')

%% Plotting first 9 dog images

tiledlayout(3,3)
for k = 1:9
   nexttile
   dog1 = reshape(dog(:,k),64,64);
   imshow(dog1)
end

%% Discrete wavelet transform of one dog image

X = im2double(reshape(dog(:,6),64,64));
[cA, cH, cV, cD] = dwt2(X,'haar');
% cA is the approximation using the low frequency content (think first few
% terms of Fourier Series)
% cH are all the horizontal features
% cV are all the vertical features
% cD are all the diagonal features

%% Plotting dwt results

subplot(2,2,1)
imshow(cA)
subplot(2,2,2)
imshow(cH)
subplot(2,2,3)
imshow(cV)
subplot(2,2,4)
imshow(cD)

% Note that the approximations are only 32x32, which is why they are pixelated

%% Rescaling the results of the dwt

cod_cH1 = rescale(abs(cH));
cod_cV1 = rescale(abs(cV));
cod_edge = cod_cH1+cod_cV1;

subplot(2,2,1)
imshow(cod_cH1)
subplot(2,2,2)
imshow(cod_cV1)
subplot(2,2,3)
imshow(cod_edge)
subplot(2,2,4)
imshow(X)


%% Wavelet transform

dog_wave = dc_wavelet(dog);
cat_wave = dc_wavelet(cat);

%% Apply SVD to dog and cat data

[U,S,V] = svd([dog_wave cat_wave],'econ');

%% Plot first four principal components

tiledlayout(2,2)
for k = 1:4
   nexttile
   ut1 = reshape(U(:,k),32,32);
   ut2 = rescale(ut1);
   imshow(ut2)
end

% look at them eyes and ears

%% Plot singular values

tiledlayout(2,1)
nexttile
plot(diag(S),'ko','Linewidth',2)
set(gca,'Fontsize',16,'Xlim',[0 80])

nexttile
semilogy(diag(S),'ko','Linewidth',2)
set(gca,'Fontsize',16,'Xlim',[0 80])

%% Plot right singular vectors
% We looked at the principal components U
% And we looked at the singular values S
% Now lets look at the right singular vectors V

for k = 1:3
   subplot(3,2,2*k-1)
   plot(1:40,V(1:40,k),'ko-')
   subplot(3,2,2*k)
   plot(1:40,V(81:120,k),'ko-')
end
subplot(3,2,1), set(gca,'Ylim',[-.15 0],'Fontsize',12), title('dogs')
subplot(3,2,2), set(gca,'Ylim',[-.15 0],'Fontsize',12), title('cats')
subplot(3,2,3), set(gca,'Ylim',[-.2 .2],'Fontsize',12)
subplot(3,2,4), set(gca,'Ylim',[-.2 .2],'Fontsize',12)
subplot(3,2,5), set(gca,'Ylim',[-.2 .2],'Fontsize',12)
subplot(3,2,6), set(gca,'Ylim',[-.2 .2],'Fontsize',12)