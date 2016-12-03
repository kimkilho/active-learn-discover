function [KDE] = util_updateKDE6(data,truelabels,KDE,tempFlag)
% Update a KDE model, folding in new instances to the model.
% function [KDE] = util_updateKDE6(data,truelabels,KDE,tempFlag)
% 
% Input: 
%  data: [ninstances x ndim] matrix.
%  truelabels: ninstances x 1 vector.
%  KDE:  Current KDE structure.
%  tempFlag: Temporarly skip merge operation for this update. Optional, default = 0.
%
% Output:
%   KDE: Updated KDE model.
%
% History:
%   V5: Caching.
%   V6: Internal caching, kernel caching.
%
% Note:
%   Multi-add and kernel caching probably don't work together.

Nk_max = KDE.Nk_max;
sigma  = KDE.sigma;
[Nd,Nx] = size(data);

assert(Nx == KDE.Nx);   %KDE & data dimensions match.
assert(isvector(truelabels)); %Labels should be vector.
assert(numel(truelabels)==size(data,1)); %Num labels match size of data.

if(nargin<4),
    tempFlag = false;
end

if(Nd>1) %Vector update
    clist = unique(truelabels);
else %Scalar update. Worth having separation because unique is slow.
    clist = truelabels;
end

for c = clist %Iterate over labels to add.
    idxK = find(truelabels==c); %Points to add for this label.
    
    if(numel(idxK)>1) %Recurse if more than one.
        for i = idxK'
            [KDE] = util_updateKDE6(data(i,:), c, KDE);
        end
        continue;
    end

    K_c = KDE.K_c;
    
    newidx = size(K_c{c,1},1)+1;    %New number w/in this class.
    K_c{c,2} = K_c{c,2}+1;          %Increment prior fro this class.
    
    %if(~tempFlag && newidx==Nk_max+1) %Skip MMat build if tempFlag or newidx < Nk_max. rong! This rebuilds every time :-(. 
    if(~tempFlag &&  newidx==Nk_max+1 && isempty(KDE.CM{c})) %Skip MMat build if tempFlag or newidx < Nk_max.
        [KDE.CM{c},KDE.MM{c}] = util_buildMergeMatrix(K_c{c,1}); %Build a merge matrix if we have hit celing.        
        disp('>> Build Merge Matrix <<');
    end
    
    if(~tempFlag && (newidx>=Nk_max+1))%Add and merge if newidx >= Nk_max && tempFlag off.
        %Now will need to do merging.
        K_c{c,1}{newidx,1} = data(idxK,:);
        if(KDE.covType==1)
            K_c{c,1}{newidx,2} = sigma;
        else
            K_c{c,1}{newidx,2} = sigma*eye(Nx);
        end
        [K_c{c,1}{newidx,3}] = 1/newidx;
        for i = 1 : newidx-1, K_c{c,1}{i,3} = K_c{c,1}{i,3}*(newidx-1)/newidx; end %Update old weights.
        K_c{c,1}{newidx,4} = cholcov(K_c{c}{newidx,2}); %Cache decomposition. 
        K_c{c,1}{newidx,5} = sum(log(diag(K_c{c}{newidx,4}))); %Cache factor. logSqrtDetSigma.
        
        %HT2 = HT;
        %if(nargin>8), 
        %    HT(idxK,:) = [idxK, c, newidx];
        %end
        %[K_c{c,1},CM{c},MM{c},idx] = util_fuseKernels(K_c{c,1},CM{c},MM{c});
        [KDE.K_c{c,1},KDE.CM{c},KDE.MM{c},indicies] = util_fuseKernels1d(K_c{c,1},KDE.CM{c},KDE.MM{c},Nk_max); %V1c also updates the cholcov/K cache.
        KDE.K_c{c,2} = KDE.K_c{c,2} + 1; %Because this doesn't get incremented in the above.
        
        KDE.updatedKernelList{c} = unique([KDE.updatedKernelList{c}, indicies(1):Nk_max]);
        
%         if(nargin>8)
%             udidx1 = (HT(:,2) == c) & (HT(:,3) == idx(1));
%             udidx2 = (HT(:,2) == c) & (HT(:,3) == idx(2));
%             shidx1 = (HT(:,2) == c) & ((HT(:,3) > idx(1)) & (HT(:,3) < idx(2))); %Move these down one kernel index.
%             shidx2 = (HT(:,2) == c) & (HT(:,3) > idx(2)); %Move these down two kernel index.
%             HT(shidx1,3) = HT(shidx1,3) - 1;
%             HT(shidx2,3) = HT(shidx2,3) - 2;
%             HT(udidx1,3) = Nk_max;
%             HT(udidx2,3) = Nk_max;
%             %if(sum(udidx1)==1 && sum(udidx2)==1), 
%             merged = {c, find(udidx1), find(udidx2), cost}; %Match any two single points to be merged. Not a point merging into an already merged cluster.a
%             %end
%         end
    end
        
    if(tempFlag || (size(K_c{c,1},1)<Nk_max+1))
        %Add normally.
        incNum = K_c{c,2}; %May be > newidx if it's tempflag.
        w      = 1/incNum;
        K_c{c,1}{newidx,1} = data;
        if(KDE.covType==1)
            K_c{c,1}{newidx,2} = sigma;
        else
            K_c{c,1}{newidx,2} = sigma*eye(Nx);
        end
        %[K_c{c,1}{1:newidx,3}] = deal(w);
        [K_c{c,1}{newidx,3}] = w;
        for i = 1 : newidx-1, K_c{c,1}{i,3} = K_c{c,1}{i,3}*(incNum-1)/incNum; end %Update old weights.
        
        K_c{c,1}{newidx,4} = cholcov(K_c{c}{newidx,2}); %Cache covariance decomposition.
        K_c{c,1}{newidx,5} = sum(log(diag(K_c{c}{newidx,4}))); %Cache factor. logSqrtDetSigma.

        KDE.updatedKernelList{c} = [KDE.updatedKernelList{c}, newidx];

        KDE.K_c = K_c;
    end   
end

KDE.updatedClassList = [KDE.updatedClassList, clist];
KDE.usedClassList  = find([KDE.K_c{:,2}]>0);
for c = KDE.updatedClassList
    %KDE.usedKernelList{c} = 1:min(KDE.K_c{c,2},Nk_max); %Assume allocated in order
    KDE.usedKernelList{c} = 1:size(KDE.K_c{c,1},1);%min(KDE.K_c{c,2},Nk_max); %Assume allocated in order
end