%--------------------------------------------------------------------------
% This is the function to call the sparse optimization program, to call the 
% spectral clustering algorithm and to compute the clustering error.
% r = projection dimension, if r = 0, then no projection
% affine = use the affine constraint if true
% s = clustering ground-truth 真实标签
% missrate = clustering error
% CMat = coefficient matrix obtained by SSC
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function [missrate,grps,CMat] = SSC(X,r,affine,alpha,outlier,rho,s)
tic
if (nargin < 6)
    rho = 1;   %建立相似矩阵中thrc函数参数K
end
if (nargin < 5)
    outlier = false;
end
if (nargin < 4)
    alpha = 20;  %admm求解参数
end
if (nargin < 3)
    affine = false;
end
if (nargin < 2)
    r = 0;      %表示不降维
end

n = max(s); %聚类数
Xp = DataProjection(X,r);%PCA降维

if (~outlier)
    CMat = admmLasso_mat_func(Xp,affine,alpha); %求解干净数据系数矩阵
    C = CMat;
else
    CMat = admmOutlier_mat_func(Xp,affine,alpha);  %求解外围点污染的情况
    N = size(Xp,2);  %降维后维度,不就是r吗
    C = CMat(1:N,:);
    toc
end

CKSym = BuildAdjacency(thrC(C,rho));
grps = SpectralClustering(CKSym,n);
missrate = Misclassification(grps,s);