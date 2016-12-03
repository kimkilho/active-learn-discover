function [A, Al, Au, pUnseen] = util_evalExpAcc3c(i,X,Y,KDE,pX,obs,opts,pC_xi)
% Absolute expected accuracy with DP unleen. Uses V5 KDE API.
% function [A, Al, Au, pUnseen] = util_evalExpAcc3(i,X,Y,KDE,pX,obs,opts,pX_Kc,pC_xi)
% Input:
%   i:  Putative label point.
%   pX_Kc:  is the dataset likelihood at the previous state of the model. Cache to speed up.
% Output:
%   ...
% 
% Note: Recently fixed bug the expected error for labeled data should have been p_{cl}(Y|X) not p_{dp}(C|X)
% 
% V3c: For big class dataset. Can optionally cap the number of classes it considers.

opts = getPrmDflt(opts, {'alpha',1,'maxConsider',20,'bigData',100});
alpha = opts.alpha;
K_c   = KDE.K_c;
seenC = find([K_c{:,2}]>0);
nSeenC = numel(seenC);
N = numel(Y);
Nc = size(K_c,1); %Because KDE is 1 more than actual size, this is actual Nc + 1.

Y2 = reshape(Y,N,1); %Ensure it's a vector.
obs = reshape(obs,N,1);
idxL = find(obs);
idxU = find(~obs);

j=1;
pC = zeros(1,Nc+1);
pC_X = zeros(N,Nc+1);
pXC = zeros(N,Nc+1);

if(Nc>opts.bigData)
    %Big data.
    [~,sidx] = sort(pC_xi(1:end-1),'descend');
    lc = sidx(1:min(opts.maxConsider,sum(pC_xi(1:Nc)>0)));
    pC_xi([lc,Nc+1]) = normalise(pC_xi([lc,Nc+1]));
    lc2 = [lc,Nc+1];
    lc = [lc,Nc];
    %disp('Doing big data');
else
    lc = find([K_c{:,2}]>0);
    lc2 = [lc,Nc+1];    
    lc = [lc,Nc];
end
A_c = zeros(numel(lc),3);

%for c = [find([K_c{:,2}]>0), Nc] %Hypothesize class.
for c = lc %Hypothesize class.    
    %Suppose it's class C. Do labeling, etc.
    Y2(i) = c;
    if(c==Nc), 
        seenC2 = [seenC, Nc];
    else
        seenC2 = seenC;
    end

    KDE2 = util_updateKDE6(X(i,:),c,KDE,true);
    
    pC(1:Nc) = [KDE2.K_c{:,2}] / (sum([KDE2.K_c{:,2}]) + alpha);
    pC(Nc+1) = alpha / (sum([KDE2.K_c{:,2}]) + alpha);
    
    pX_Y = util_inferKDE6(X,KDE2,0,1,0,1,0); %Cached update.
    %[lab,pX_Y,p,lp] = util_inferKDE3(X,K_c2,0,pX_Kc,c); %Cached update.
    %[lab2,pX_Y2,p2,lp2] = util_inferKDE3(X,K_c2); %Full update.
    %assert(all(pX_Y2(:)==pX_Y(:))); %Ensure the same.
    
    %pY_X2= normalise(pX_Y.* repmat(pC(1:end-1), N, 1),2);
    %pY_X = normalise(pX_Y.* (ones(N,1)*pC(1:end-1)),2); %Faster than repmat.
    
    pXY  = pX_Y.* (ones(N,1)*pC(1:end-1));
    pY_X = normalise(pXY,2);    %Excluding new class.
    pXC(:,1:Nc) = pXY;
    pXC(:,end)  = pX * pC(end); %Including new class.
    pC_X = normalise(pXC,2);

    if(~isempty(idxL))
        %Rl = sum(1-pC_X(idxL,Y(idxL))); %Risk against true labels.
        %ind = sub2ind(size(pC_X), idxL, Y2(idxL)');
        %Al = sum(pC_X(ind)); %Risk against true labels.
        %ind2= sub2ind(size(pY_X), idxL, Y2(idxL)); %sub2ind is slow!
        ind = (Y2(idxL)-1)*N+idxL; %Replace sub2ind
        %assert(all(ind==ind2));
        Al = sum(pY_X(ind)); %Risk against true labels.
    else
        Al = 0;
    end

    pY_Xseen = pY_X(:,seenC2);
    pC_Xseen = pC_X(:,[seenC2,Nc+1]);
    Au = sum(sum(pC_Xseen(idxU,1:end-1) .* (pY_Xseen(idxU,:))));

    A = Au + Al;
    A_c(j,:) = [A; Al; Au];
       
    j=j+1;
end

%A_i = sum(A_c .* repmat(pC_xi(pC_xi>0)', 1,3),1);
%A_i = sum(A_c .* repmat(pC_xi([seenC,Nc+1])', 1,3),1);
A_i = sum(A_c .* repmat(pC_xi(lc2)', 1,3),1);
A = A_i(1);
Al = A_i(2);
Au = A_i(3);
pUnseen = pC_Xseen(:,end);

