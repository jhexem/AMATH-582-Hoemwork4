% Clean workspace
clear all; close all; clc

%% Loading cat and dog data

load('catData.mat')
load('dogData.mat')


%% Redo Wavelet transform

dog_wave = dc_wavelet(dog);
cat_wave = dc_wavelet(cat);

%% Reapply SVD to dog and cat data

[U,S,V] = svd([dog_wave cat_wave],'econ');

%% Project onto PCA modes

feature = 20; % The number of PCA modes we are going to project onto.  We
% call these modes "features".

nd = size(dog_wave,2);
nc = size(cat_wave,2);
animals = S*V'; % projection onto principal components: X = USV' --> U'X = SV'
dogs = animals(1:feature,1:nd); % first several rows of the dog cols
cats = animals(1:feature,nd+1:nd+nc); % first several rows of the cat cols

%% Calculate scatter matrices (i.e., covariances)

md = mean(dogs,2);
mc = mean(cats,2);

Sw = 0; % within class variances
for k = 1:nd
    Sw = Sw + (dogs(:,k) - md)*(dogs(:,k) - md)';
end
for k = 1:nc
   Sw =  Sw + (cats(:,k) - mc)*(cats(:,k) - mc)';
end

Sb = (md-mc)*(md-mc)'; % between class

%% Find the best projection line

[V2, D] = eig(Sb,Sw); % linear disciminant analysis; i.e., generalized eval. prob.
[lambda, ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);

%% Project onto w

vdog = w'*dogs;
vcat = w'*cats;

%% Orient all the dogs on the lower end and all the cats on the higher end
% For orientation purposes since PatternRecAns.mat saves dogs as 0 and cats
% as 1.

if mean(vdog) > mean(vcat)
    w = -w;
    vdog = -vdog;
    vcat = -vcat;
end

%% Plot dog/cat projections (not for function)

plot(vdog,zeros(80),'ob','Linewidth',2)
hold on
plot(vcat,ones(80),'dr','Linewidth',2)
ylim([0 1.2])

%% Find the threshold value

sortdog = sort(vdog);
sortcat = sort(vcat);

t1 = length(sortdog); % start on the right
t2 = 1; % start on the left
while sortdog(t1) > sortcat(t2) 
    t1 = t1 - 1;
    t2 = t2 + 1;
end
% ^ go past each other

threshold = (sortdog(t1) + sortcat(t2))/2; % get the midpoint

%% Plot histogram of results


tiledlayout(1,2)

nexttile
histogram(sortdog,30); hold on, plot([threshold threshold], [0 10],'r')
set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('dog')

nexttile
histogram(sortcat,30); hold on, plot([threshold threshold], [0 10],'r')
set(gca,'Xlim',[-3 4],'Ylim',[0 10],'Fontsize',14)
title('cat')

% And this is called training
% We are going to save all of this in a function called dc_trainer.m