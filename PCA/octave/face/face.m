clear;
close all;
clc;


%%===========================
%   the data is from coursera
%   there are totally 5000 faces
%	and each face is 32*32 = 1024 dimision
%	our target is to reduce this to 100 dimision
%============================
K = 100;
load('faces.mat');
%size(X)

%  Before running PCA, it is important to first normalize X
[X_norm, mu, sigma] = featureNormalize(X);


%%===========================
%  display the origin faces
%  why norm don't affect Displaying ?
%============================
displayData(X_norm(1:36,:));
fprintf('Program paused. Displaying Origin First 36 faces\n');
title('origin face');
pause;



fprintf('Computing PCA ....\n');
% perform pca on normalized data
[m n] = size(X_norm);
Covar = X_norm' * X_norm / m;
[U, S, V] = svd(Covar);

% pca main step is done
% U is components

figure;
displayData(U(:,1:36)');
fprintf('Program paused. Displaying founded first 36 components\n');
title('first 36 components');
pause;

% the reduced values
% no meaning
figure;
U_reduce = U(:, 1:K);
Z = X * U_reduce;
displayData(Z(1:36, :));
title('projected data');
fprintf('Program paused. Displaying project data\n');
pause;

% recover from Z
figure;
X_rec = ( U_reduce * Z')';
displayData(X_rec(1:36, :));
title('recovered face');
fprintf('Program paused. Displaying recover face with 100 dimision\n');
pause;


close all;

