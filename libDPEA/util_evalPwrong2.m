function [idx,pidx,KDE,pC_X] = util_evalPwrong2(X,KDE,obs,opts,pX)
%% Active learning query criteria - select the most likely to be wrong under DP/KDE assumption. (My implementation of Haines'11 BMVC)
% function [idx,pidx,pX_Kc,pC_X] = util_evalPwrong(X,KDE,obs,opts,pX,pX_Kc,c)
% Input: 
%   X:      Data
%   KDE:    KDE Model.
%   obs:    Observation bitvect.
%   pX:     Prior density
%   opts
%       .alpha:     DP concentration (def 1)
%       .usePrior:	Use prior (def 1, boolean)
%       .sampQuery: 1. Sample pidx to get idx. 0. Maximize pidx.  (def 0)
%
% Output: 
%   idx:    Selected point
%   pidx:   Query distribution.
%   pX_Kc:  Updated likelihood for caching.
%   pC_X:   DP Posterior.
% 
% V2: Use with KDE API V6.

opts = getPrmDflt(opts, {'alpha',1,'usePrior',1,'sampQuery',0,'kernelCache',2});
N = size(X,1);
if(opts.kernelCache==2)
    [pX_Y,labels,~,KDE] = util_inferKDE6(X,KDE,0,1,1,1,1); %Use and store.
elseif(opts.kernelCache==1)
    [pX_Y,labels,~,KDE] = util_inferKDE6(X,KDE,0,1,1);
else
    [pX_Y,labels,~,KDE] = util_inferKDE6(X,KDE,0);
end

if(opts.usePrior)
    pC_X          = pX_Y .* repmat(([KDE.K_c{:,2}]) / (sum([KDE.K_c{:,2}]) + opts.alpha), N,1);
    pC_X(:,end+1) = pX * opts.alpha / (sum([KDE.K_c{:,2}]) + opts.alpha);
else
    pC_X = [pX_Y, pX];
end

pC_X = normalise(pC_X,2); %Work out p(Classes, Unseen | Dataset).

ind = sub2ind(size(pC_X), 1:N, labels'); %Find the linear index of the MAP coords for each.

%1-p(correct) method.
pWrong_X = 1-pC_X(ind); 

%\sum_i[p(wrong_i)] method) should be the same.
%pCwrong_X = pC_X;
%pCwrong_X(ind) = 0;
%pWrong_X1 = sum(pCwrong_X,2); %
%fprintf(1,'Avg diff: %1.2f\n', sum(abs(pWrong_X1-pWrong_X2')));

pidx = pWrong_X;
pidx(obs) = 0;
if(opts.sampQuery) %Sample from query dist.
    if(all(pidx==0)), 
        disp('Fixing pidx'); 
        pidx(~obs)=1/sum(~obs); 
    end
    idx = sum(cumsum(normalise(pidx))<rand) + 1;
    assert(~obs(idx)); %Sanity check we didn't pick an already picked item.
else %Maximize query dist.
    [~,idx] = max(pidx);
end
