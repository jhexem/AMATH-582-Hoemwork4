% Clean workspace
clear all; close all; clc

%% Train model

load('catData.mat')
load('dogData.mat')

dog_wave = dc_wavelet(dog);
cat_wave = dc_wavelet(cat);

feature = 20;
[U,S,V,threshold,w,sortdog,sortcat] = dc_trainer(dog_wave,cat_wave,feature);

%% Load and view from the test set
% Random labeled(hidden) dogs and cats

load('PatternRecAns')

tiledlayout(3,3)
for k = 1:9
   nexttile
   test = reshape(TestSet(:,k+5),64,64);
   imshow(test)
end

%% Classify test data

TestNum = length(TestSet(1,:));
Test_wave = dc_wavelet(TestSet); % wavelet transform
TestMat = U'*Test_wave; % PCA projection
pval = w'*TestMat; % Project onto w vector

%% Check pval against threshold

% Cat = 1, dog = 0 for ResVec
ResVec = (pval > threshold) % results for test set

%% Checking performance
% let's see how many we got correct

% 0s are correct and 1s are incorrect
err = abs(ResVec - hiddenlabels)
errNum = sum(err);
sucRate = 1 - errNum/TestNum;

%% Checking errors

k = 1;
TestNum = length(pval);

tiledlayout(1,2)
for j = 1:TestNum
   if ResVec(j) ~= hiddenlabels(j)
      S = reshape(TestSet(:,j),64,64);
      nexttile
      imshow(S)
      k = k+1;
   end
end

%% Classifying your pets

% Clear pval
pval = [];

close all

% Read in image
%I = imread('Rory.jpeg');
%I = imread('Kaitlynn_pet.jpg');
%I = imread('AlexJ_pet.jpg');
%I = imread('Ali_Cartoon.webp');
%I = imread('Kaitlynn_pet3.jpg');
I = imread("Astro.jpg");
%I = imread('SeanCat.jpg');
imshow(I)

% Convert to grayscale and resize
I = rgb2gray(I);
I = im2double(I);
I = imresize(I,[64,64]);
I = reshape(I,64*64,1);

%  Classify the image
I_wave = dc_wavelet(I); % wavelet transform
IMat = U'*I_wave; % PCA projection
pval = w'*IMat;

if pval > threshold
    disp('Cat')
else
    disp('Dog')
end
