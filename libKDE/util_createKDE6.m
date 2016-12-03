function [KDE] = util_createKDE6(data,labels,covType,sigma,Nk_max,Nc)
% Creates a KDE data structure.
% function [KDE] = util_createKDE5(data,labels,covType,sigma,Nk_max,Nc)
%
% Input:
%   data:   [Ninstance x Ndim] matrix.
%   labels: [Ninstance x 1] vector. (default: all same class)
%   covType: 'full' or 'diag'       (default: 'full')
%   sigma:  Initial covariance. Scalar. (default: 1)
%   Nk_max: Max number of kernels to use. (default: unlimited)
%   Nc:     Number of classes to allocate space for. (default: |unique(labels)|).
% 
% Output: 
%   KDE:    Kernel Density Estimate structure.
%   KDE.Nx: Dimension of data.
%   KDE.Nc: Max number of classes (just for bookkeeping).
%   KDE.K_c:
%       one row per class. 
%       col1: class kernels
%       col2: class prior.
%   KDE.K_c{:,1} Kernels: 
%       K_c{:,1}{1}: Gaussian mean.
%       K_c{:,1}{2}: Gaussian covariance.
%       K_c{:,1}{3}: Gaussian weight.
%       K_c{:,1}{4}: Cached Gaussian cov cholskey decomposition.
%       K_c{:,1}{4}: Cached Gaussian prefactor.
%   KDE.CM: Cost matrix cell array. One entry per class.
%   KDE.WM: Merge matrix cell array. One entry per class.
%
% Todo: the 'HT' logging stuff is broken for the moment.

global VERBOSE
if(nargin<2), %Everything in the same class if not specified.
    labels = ones(size(data,1),1); 
end

if(nargin<4), 
    sigma = 1; 
end

KDE.sigma = sigma;

if(nargin>=5)
    KDE.Nk_max = Nk_max;
else
    KDE.Nk_max = inf;
end

if(nargin < 3 || strcmp(covType,'full'))
    KDE.covType = 2;
    if(VERBOSE>0), disp('Assuming full covariance'); end
else
    KDE.covType = 1;
    if(VERBOSE>0), disp('Assuming UD covariance'); end
end

assert(size(data,1)==numel(labels));
assert(isvector(labels)); 

%Assume number of classes is number of unique labels. This is just for accounting purposes.
if(nargin<6)    
    Nc  = numel(unique(labels));
end

KDE.Nc = Nc;
K_c = cell(Nc,2);
[K_c{:,2}] = deal(0);
[Nd,Nx]  = size(data);
KDE.Nx = Nx;

if(nargin<5 || all(hist(labels,1:Nc)<=Nk_max))
    %% One shot creation if we are not at kernel limit.
    W = zeros(1,Nc);
    for c = 1 : Nc
        K_c{c,2} = sum(labels==c);
        W(c) = 1/K_c{c,2}; %Weight per point.
        %For now just keep count of each class.
    end
    KDE.CM = cell(1,Nc);
    KDE.MM = cell(1,Nc);
    for i = 1:Nd %Add kernels for each point in turn.
        c = labels(i);
        insert = size(K_c{c},1)+1;
        K_c{c}{insert,1} = data(i,:);
        K_c{c}{insert,3} = W(c);
        if(KDE.covType==1)
            K_c{c}{insert,2} = sigma;
            K_c{c}{insert,4} = cholcov(K_c{c}{insert,2}*eye(Nx)); %Cache decomposition. % Abit hacky having this here...
            K_c{c}{insert,5} = sum(log(diag(K_c{c}{insert,4})));  %Cache factor. logSqrtDetSigma.
        elseif(KDE.covType==2)
            K_c{c}{insert,2} = sigma*eye(Nx);
            K_c{c}{insert,4} = cholcov(K_c{c}{insert,2}); %Cache decomposition.
            K_c{c}{insert,5} = sum(log(diag(K_c{c}{insert,4}))); %Cache factor. logSqrtDetSigma.
        end           
    end
    KDE.K_c = K_c;
else
    %% If we are over kernel limit, start by selecting a random Nk_max instances from each class and do one shot creation with that subset.
    lclass = cell(1,Nc); 
    for c = 1 : Nc
        tmp = find(labels==c)';
        ridx = randperm(numel(tmp));
        lclass{c} = tmp(ridx(1:min(numel(tmp),Nk_max)));
    end
    linst = [lclass{:}];
    KDE = util_createKDE6(data(linst,:),labels(linst),'full',sigma,Nk_max,Nc);
    %% Now iteratively fold in the remaining datapoints.
    bitsToAdd = true(1,Nd);
    bitsToAdd(linst) = false;
    if(exist('ticStatus','file') && Nd>1000), ticId = ticStatus; end
    for i = find(bitsToAdd)
        [KDE] = util_updateKDE6(data(i,:),labels(i),KDE);
        if(Nd>1000 && exist('ticStatus','file'))
            tocStatus(ticId,i/Nd);
        end
    end    

end

%% Stuff for caching.
KDE.updatedClassList = unique(labels(:))';
KDE.usedClassList = unique(labels(:))';
KDE.storeLikCache = 1; %Default.
KDE.storeKernelCache = 1;
KDE.kernelCache = cell(1,Nc);
KDE.updatedKernelList = cell(1,Nc);
KDE.usedKernelList = cell(1,Nc);
for c = KDE.updatedClassList 
    KDE.updatedKernelList{c} = 1:size(KDE.K_c{c,1},1); %All kernel created updated.
    KDE.usedKernelList{c} = 1:min(KDE.K_c{c,2},KDE.Nk_max); %Assume allocated in order
end