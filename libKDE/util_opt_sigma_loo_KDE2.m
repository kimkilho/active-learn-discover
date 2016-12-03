function [S_c,llh_s,lSigma] = util_opt_sigma_loo_KDE2(K,priorplus)
% LOO Cross validation to determine \sigma for a KDE model.
% function [S_c,llh_s,lSigma] = opt_sigma_loo_KDE2(K_c)
% Input: 
%   K:   kernel structure from a single class KDE.
% Output:
%   S_c: Optimal \Sigma.
%   lSigma: List of sigma evaluated.
%   llh:    Resulting loo-llh for ^.

disp('Optimizing KDE \Sigma');
lSigma = logspace(-2,1,50); %Range of sigma to evaluate.
llh_s  = zeros(size(lSigma));
Nx = numel(K{1,1});
Nd = size(K,1);

% Make the n^2 sized distance matrix. Upper triangular.
%fprintf(1,'Building initial dist mat\n');
distMat = zeros(Nd,Nd);
for d1 = 1 : Nd
    for d2 = d1+1 : Nd
        %CPU time *HERE* - optimize?
        distMat(d1,d2) = sum((K{d1,1}-K{d2,1}).^2);
    end
end

%fprintf(1,'Sigma = 1:50 ');
k=1;
for s = lSigma
    llh_s(k) = looKDE2(distMat,s,Nx);
    k=k+1;
    %fprintf('..%d',k);
end
[~,idx] = max(llh_s);
if(nargin>1)
    idx=idx+priorplus;
end
S_c = lSigma(min(idx,numel(lSigma)));


function llh = looKDE1(distmat,sigma,Nx)
%% function llh = looKDE1(distmat,sigma,Nx)
% Compute LOO estimate for generalization of p(X|\Sigma)
Nd = size(distmat,1);
s1 = -0.5/sigma^2;
s2 = -(Nx/2)*log(2*pi*sigma^2);
llh = 0;
for d = 1 : Nd
    %CPU time *HERE* - optimize?
    llh = llh + s2+log(sum(exp(s1*distmat(d,d+1:end)))+sum(exp(s1*distmat(1:d-1,d))));
end
function llh = looKDE2(distmat,sigma,Nx)
%% function llh = looKDE1(distmat,sigma,Nx)
% Compute LOO estimate for generalization of p(X|\Sigma)
Nd = size(distmat,1);
s1 = -0.5/sigma^2;
s2 = -(Nx/2)*log(2*pi*sigma^2);
smat = s1*distmat;
llh = 0;
for d = 1 : Nd
    %CPU time *HERE* - optimize?
    llh = llh + s2+log(sum(exp(smat(d,d+1:end)))+sum(exp(smat(1:d-1,d))));
end