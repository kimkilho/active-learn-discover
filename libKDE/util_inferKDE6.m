function [np2,labels,p2,KDE,lp2] = util_inferKDE6(data,KDE,use_prior,useLikCache,storeLikCache,useKernelCache,storeKernelCache)
% Inference the likelihood / class of new data under a specified KDE.
% function [np2,labels,p2,KDE,lp2] = util_inferKDE6(data,KDE,use_prior,useLikCache,storeLikCache,useKernelCache,storeKernelCache)
%
% Input:
%   data:        [ninst x ndim] array.
%   KDE:         KDE data structure holding the model.
%   use_prior:   Boolean flag to use prior or not for inference. Default: Yes.
%   oldnp2:      Old p(x|y) cache. Optional.
%   cud:         Updated y elements, cache the others with ^^.  Optional.
%
% Output:
%   np2:      Likelihood. p(x|c) for each data item and class. unnormalized.
%   labels:   MAP estimate c|x  for each data item. 
%   p2:       Posterior. p(c|x) for each data item and class. normalized.
%   lp2:      log(p(c|x)) for each data item and class.
%   Note:     p2 uses only seen classes, the others use all the class dimensions allocated by the KDE structure. 
%
% History:
%   V6: Make likelihood caching internal.
%   V5: Various caching, for efficiency.
%   V4: Big KDE model.
%   V3: No observed, truelabels.
%
% TODO: Assuming full covariance ATM.

[Nd,Nx] = size(data);

N_c = KDE.Nc;
K_c = KDE.K_c;

assert(Nx == KDE.Nx); %Data dimension must match KDE dimension.

if(nargin>4 && useLikCache && isfield(KDE,'likCache'))
    clist = KDE.updatedClassList;
    np2 = KDE.likCache;
    assert(size(np2,2)==N_c); %Ensure data size matches cache size.
    assert(size(np2,1)==Nd);
    np2(:,clist) = 0;
else
    %clist = 1:N_c;
    clist = KDE.usedClassList;
    np2 = zeros(Nd, N_c);
end

K = Nx*log(2*pi)/2; %Gaussian prefactor.

if(KDE.storeKernelCache && nargin > 5 && useKernelCache && isfield(KDE,'kernelCache'));
    %% Update likelihood with kernel cache.
    for c = clist
        
        if(isempty(KDE.kernelCache{c})) %No kernel cache yet.
            kk = zeros(Nd,KDE.Nk_max+1);
        else
            kk = KDE.kernelCache{c}; %Cache for class c.
        end                
        klist = KDE.updatedKernelList{c};
        
        for k = klist %Only check data against updated kernels.
            X0 = bsxfun(@minus,data,K_c{c,1}{k,1}); %X-Mu.
            xRinv = X0 / K_c{c,1}{k,4};             %Use cached cholcov.
            quadform = sum(xRinv.^2, 2);
            %kk(:,k) = K_c{c,1}{k,3} * exp(-0.5*quadform - K_c{c,1}{k,5} - K);
            kk(:,k) = exp(-0.5*quadform - K_c{c,1}{k,5} - K);
        end
        
        % Lik doesn't changed for cached kernels, but weights can, so multiply in the weights later.
        np2(:,c) = sum( kk(:,KDE.usedKernelList{c}) .* (ones(Nd,1)* [KDE.K_c{c,1}{:,3}]), 2); %
        
        if(KDE.storeKernelCache && nargin>6 && storeKernelCache)
            KDE.kernelCache{c} = kk;
            KDE.updatedKernelList{c} = [];            
        end
        
    end
else
    %% Update likelihood classic way
    for c = clist
        for k = 1 : size(K_c{c,1},1) %Evaluate the likelihod of all the data (vector) against each kernel.
            %Old, matlab gaussian PDF. Slow.
            %covs = {K_c{c,1}{:,2}};
            %np2(:,c) = np2(:,c) + K_c{c,1}{k,3} * mvnpdf(data, K_c{c,1}{k,1}, K_c{c,1}{k,2});

            X0 = bsxfun(@minus,data,K_c{c,1}{k,1}); %X-Mu.
            xRinv = X0 / K_c{c,1}{k,4};             %Use cached cholcov.
            quadform = sum(xRinv.^2, 2);
            np2(:,c) = np2(:,c) + K_c{c,1}{k,3} * exp(-0.5*quadform - K_c{c,1}{k,5} - K); %Used cached logSqrtdetSigma.
        end
    end
end

if(KDE.storeLikCache)
    if(nargin > 4 && storeLikCache)
        KDE.likCache = np2;
        KDE.updatedClassList = [];
    end
end


%Sigma = K_c{c,1}{k,2};            
%[R,err] = cholcov(Sigma,0);
%xRinv = X0 / R;
%logSqrtDetSigma = sum(log(diag(R)));
%quadform = sum(xRinv.^2, 2);
%y(:,c) = y(:,c) + K_c{c,1}{k,3} * exp(-0.5*quadform - logSqrtDetSigma - d*log(2*pi)/2);
%%np2(:,c) = np2(:,c) + K_c{c,1}{k,3} * exp(-0.5*quadform - logSqrtDetSigma - Nx*log(2*pi)/2);


if(nargout>1) %Only do this extra stuff if asked for posteriors.
    % For some reason prior dominates too much on synthetic spiral + spots
    % dataset! :-(.
    if(nargin<3)
        %use_prior=true;
        use_prior=false;
    end
    if(use_prior)
        pr = [K_c{:,2}]; %Don't have to normalize for the next use.
        p2=normalise(np2.*repmat(pr,[Nd,1]),2); %Replace this with a multiplyout.
    else
        %pr = ones(1,N_c);
        p2=normalise(np2,2);  %Quicker no repmat.
    end
    allzeros = (sum(p2,2)==0);
    seenC = [K_c{:,2}]>0;
    
    %Fix all zeros underflow?
    if(any(allzeros))
        fprintf(1,'KDE: Warning fixing %d zeros\n',sum(allzeros));
        p2(allzeros,seenC) = 1/sum(seenC);%0.5;
    end
    [~,labels] = max(p2,[],2);
    
    %labels = labels';
    lp2 = log(p2); %Probs -> logprobs.

    p2 = p2(:,seenC);
end



    