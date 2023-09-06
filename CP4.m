%%% Clean workspace
clear all; close all; clc

numSingVals = 15;
%{
allAvgs = zeros(1, 196);
for c = 1:49
c

sucRateList = zeros(1, 3);
for b = 0:9
num0 = b;
%}
%%% Load training data

load('CP4_training_labels.mat')
load('CP4_training_images.mat')

num0 = 2;
%num1 = 1;

%% Reshape training_images.m to 784x30000 in order to be consistent with the
% codes from the lecture.

training_images = reshape(training_images, [784, 30000]);
%imshow(reshape(training_images(:, 29999), [sqrt(784), sqrt(784)]))

%% Projecting onto principal components

% Conduct the wavelet transform on the entire training dataset.  To make it
% reusable you can write it as a function very similar (almost exactly the
% same) as dc_wavelet.m from the lecture.  For MATLAB you will need to
% include the function at the end of the code.

%numWave = num_wavelet(training_images);

load("Training_DWT.mat")
numWave = Training_DWT;

%% Find the SVD of the transformed data just like in Week7_LDA.m

[U,S,V] = svd(numWave, 'econ');

%% Plot singular values (include in report, but not in gradescope submission)
%{
sVals = diag(S);
plot(sVals, ".r", "MarkerSize",10)
xlabel('Singular Values')
ylabel('Size of Each Value')
%}
%% How many features (i.e., singular values) should we use?  Save this as
% A1.  Hint: it's less than what we used in the lecture

A1 = numSingVals; % 1x1. The number of PCA modes we are going to project onto.

%% Project onto the principal components just like we did in Week7_LDA.m
% Restrict the matrix U to that of the feature space (like we did in 
% dc_trainer.m).  Save this as A2

numImages = length(training_labels);
newNums = S*V';
U = U(:,1:A1);
nums = newNums(1:A1,1:numImages);

A2 = U; % 196x15

%% Pick two numbers to train/test on.  Use 0 and 1 for the autograder.

% This is going to be quite different from what we did in the lectures.  In
% the lecture we had two datasets with dogs and cats.  Here everything is
% jumbled up so we need to separate them out.  Separate all the training 
% images of 0's and 1's using the training labels.  Hint: a for loop and
% some if statements should be sufficient.

ZerosIdx = [];
OnesIdx = [];

for j = 1:length(training_labels)
    if training_labels(j) == num0
        ZerosIdx = [ZerosIdx, j];
    else
    %elseif training_labels(j) == num1
        OnesIdx = [OnesIdx, j];
    end
end

numZeros = length(ZerosIdx);
numOnes = length(OnesIdx);

imageZeros = nums(:, ZerosIdx);
imageOnes = nums(:, OnesIdx);

%% Calculate the within class and between class variances just like in 
% Week7_LDA.m.  Save these as A3 and A4.

avgZeros = mean(imageZeros, 2);
avgOnes = mean(imageOnes, 2);

withinVar = 0; % within class variances
for k = 1:numZeros
    withinVar = withinVar + (imageZeros(:,k) - avgZeros)*(imageZeros(:,k) - avgZeros)';
end
for k = 1:numOnes
    withinVar =  withinVar + (imageOnes(:,k) - avgOnes)*(imageOnes(:,k) - avgOnes)';
end

betweenVar = (avgZeros - avgOnes)*(avgZeros - avgOnes)'; % between class

A3 = withinVar; % 15x15
A4 = betweenVar; % 15x15

%% Find the best projection line just like in Week7_LDA.m.  Save the
% normalized projection line w as A5

[genEigVecs, genDiagMatrix] = eig(betweenVar,withinVar); % linear disciminant analysis; i.e., generalized eval. prob.
[maxLambda, idxMaxLambda] = max(abs(diag(genDiagMatrix)));
maxEigVec = genEigVecs(:,idxMaxLambda);
w = maxEigVec/norm(maxEigVec,2);

A5 = w; % 15x1

%% Project the training data onto w just like in Week7_LDA.m

vZeros = w' * imageZeros;
vOnes = w' * imageOnes;

%% Plot Projections
%{
plot(vZeros,zeros(numZeros),'ob','Linewidth',2)
hold on
plot(vOnes,ones(numOnes),'dr','Linewidth',2)
ylim([0 1.2])
%}
%% Find the threshold value just like in Week7_LDA.m.  Save it as A6

sortZeros = sort(vZeros);
sortOnes = sort(vOnes);

diff = mean(sortOnes) - mean(sortZeros);

if diff > 0
    t1 = length(sortZeros); % start on the right
    t2 = 1; % start on the left

    while sortZeros(t1) > sortOnes(t2) 
        t1 = t1 - 1;
        t2 = t2 + 1;
    end

    threshold = (sortZeros(t1) + sortOnes(t2))/2;
else
    t1 = 1; % start on the left
    t2 = length(sortOnes); % start on the right

    while sortOnes(t2) > sortZeros(t1) 
        t1 = t1 + 1;
        t2 = t2 - 1;
    end

    threshold = (sortZeros(t1) + sortOnes(t2))/2;
end

A6 = threshold; % 1x1

%% Classify test data

load('CP4_test_labels.mat')
load('CP4_test_images.mat')

% Reshape test_images.m to 784x5000 in order to be consistent with the
% codes from the lecture.

test_images = reshape(test_images, [784, 5000]);

% From the test set pick out the 0's and 1's without revealing the labels.
% Save only the images of 0's and 1's as a new dataset and save the
% associated labels for those exact 0's and 1's as a new vector.

testNumsIdx = [];
testNumsVal = [];

for j = 1:length(test_labels)
    if test_labels(j) == num0
        testNumsIdx = [testNumsIdx, j];
        testNumsVal = [testNumsVal, 0];
    else
    %elseif test_labels(j) == num1
        testNumsIdx = [testNumsIdx, j];
        testNumsVal = [testNumsVal, 1];
    end
end

sizeTestNums = length(testNumsIdx);
imageTestNums = test_images(:, testNumsIdx);

% Wavelet transform:  you can just use the same function you did for the
% training portion.

waveTestNums = num_wavelet(imageTestNums);
%{
load("Test_DWT.mat")
waveTestNums = Test_DWT;
%}
% Project the test data onto the principal components just like in
% Week7_Learning.m
% Save the results in a vector (just like in Week7_Learning.m) and save it
% as A7.

pcaProjTestNums = U' * waveTestNums;
vTestNums = w' * pcaProjTestNums;

testResults = vTestNums > threshold;

A7 = double(testResults); % 1x1062

%%% Checking performance just like we did in Week7_Learning.m.  If you did
%%% everything like I did (which may or may not be optimal), you should
%%% have a success rate of 0.9972.

err = abs(testResults - testNumsVal);
errNum = sum(err);
succNum = sizeTestNums - errNum;
sucRate = 1 - errNum / sizeTestNums;

%% For report only, not for the autograder:  Now write an algorithm to
%%% classify all 10 digits.  One way to do this is by using the "one vs all
%%% " method; i.e., loop through the digits and conduct LDA on each digit
%%% vs. all the other digits.

load('CP4_test_labels.mat')
load('CP4_test_images.mat')

test_images = reshape(test_images, [784, length(test_images)]);

testNumsIdx = [];
testNumsVal = [];

for j = 1:length(test_labels)
    if test_labels(j) == num0
        testNumsIdx = [testNumsIdx, j];
        testNumsVal = [testNumsVal, 0];
    else
    %elseif test_labels(j) == num1
        testNumsIdx = [testNumsIdx, j];
        testNumsVal = [testNumsVal, 1];
    end
end

sizeTestNums = length(testNumsIdx);
imageTestNums = test_images(:, testNumsIdx);

waveTestNums = num_wavelet(imageTestNums);
pcaProjTestNums = U' * waveTestNums;
vTestNums = w' * pcaProjTestNums;

if diff > 0
    testResults = vTestNums > threshold;
else
    testResults = vTestNums < threshold;
end

err = abs(testResults - testNumsVal);
errNum = sum(err);
succNum = sizeTestNums - errNum;
sucRate = 1 - errNum / sizeTestNums;
%{
sucRateList(b+1) = sucRate;
end
allSucRates = sucRateList
totalSucRate = mean(allSucRates)
%}
%allAvgs(c) = totalSucRate;
%end

%% Put any helper functions here

%plot(4 * (1:49), allAvgs(1:49), 'r.', "MarkerSize", 8)

%% 

sucRateVals15 = [0.9580, 0.9792, 0.9336, 0.9212, 0.9128, 0.9148, 0.9578, 0.9614, 0.9312, 0.9274];
meanErr15 = mean(sucRateVals15);
sucRateVals50 = [0.9686, 0.9822, 0.9454, 0.9292, 0.9332, 0.9286, 0.9660, 0.9660, 0.9416, 0.9388];
meanErr50 = mean(sucRateVals50);
sucRateVals150 = [0.9694, 0.9860, 0.9580, 0.9402, 0.9480, 0.9436, 0.9652, 0.9576, 0.9432, 0.9498];
meanErr150 = mean(sucRateVals150);

%% My Numbers

myVal = 2;
myNum = imread('two.jpg');
myNum = rgb2gray(myNum);
myNum = im2double(myNum);
myNum = imresize(myNum,[28,28]);
imshow(myNum)
myNum = reshape(myNum,784,1);

waveTestNums = num_wavelet(myNum);
pcaProjTestNums = U' * waveTestNums;
vTestNums = w' * pcaProjTestNums;

if diff > 0
    testResults = vTestNums > threshold;
else
    testResults = vTestNums < threshold;
end

testResults

%% 



function numData = num_wavelet(numfile) 
% Input: dog/cat matrix
% Output: edge detection

    [m,n] = size(numfile); % 784x30000
    pxl = sqrt(m); % cus images are square
    nw = m/4; % wavelet resolution cus downsampling
    numData = zeros(nw,n);
    
    for k = 1:n
        X = im2double(reshape(numfile(:,k),pxl,pxl));
        [~,cH,cV,~]=dwt2(X,'haar'); % only want horizontal and vertical
        num_cH1 = rescale(abs(cH)); % horizontal rescaled
        num_cV1 = rescale(abs(cV)); % vertical rescaled
        num_edge = num_cH1+num_cV1; % edge detection
        numData(:,k) = reshape(num_edge,nw,1);
    end
end
