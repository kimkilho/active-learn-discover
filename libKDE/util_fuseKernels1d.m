function [K_c,CM,MM,indicies,cost,lUpdated] = util_fuseKernels1d(K_c,CM,MM,Nk_max)
% Selects a kernel pair and fuses them.
% function [K_c,CM,MM,indicies] = fuseKernels1c(K_c,CM,MM,Nk_max)
% History:
%	V1c: Also does some caching for the use of calling routines.
% Input: 
%   K_c: KDE structue.
%   CM: Cost matrix.
%   MM: Merge matrix. 
%   Note: Temporary kernel assumed last.
% Output:
%   K_c: New KDE structure, one kenel less than input.
% Notes:
%   For input, temporary kernel assumed last.
%   K_c should have K_max+1, and CM, MM need to be temporarily K_max+1.
%
% V1d: Include list of updated kernels.

CM2 = zeros(size(CM)+1);
MM2 = cell(size(CM)+1);
CM2(1:Nk_max,1:Nk_max) = CM;
MM2(1:Nk_max,1:Nk_max) = MM;
%Expand the merge and cost matricies to include the temporary element.
for k1 = 1 : Nk_max
    for k2 = Nk_max+1
        %MM2{k1,k2} = util_GaussMerge({K_c{k1,:}},{K_c{k2,:}});
        MM2{k1,k2} = util_GaussMerge(K_c(k1,:),K_c(k2,:));
        CM2(k1,k2) = K_c{k1,3}*util_GaussKLD(K_c(k1,:),MM2{k1,k2}) + K_c{k2,3}*util_GaussKLD(K_c(k2,:),MM2{k1,k2});
    end
end
CM2(tril(ones(size(CM2)))>0) = inf;
cost=min(CM2(:)); %Return the cost paid for debug.
[k1,k2] = find(CM2==min(CM2(:)));   %Find the min cost to merge. %K1 & K2 are the pair to merge.
if(numel(k1)>1)
    disp('Warning multiple points at same distance');
    k1=k1(1);
    k2=k2(1);
end
indicies=[k1,k2];
indicies = sort(indicies,'ascend');
s1 = indicies(1); s2=indicies(2);

%Updated kernels. For now conservatively say everything past the first one chosen for fusion is updated because their order can change :-/.
lUpdated = s1+1:Nk_max;

% Update the kernel list to exclude the deleted ones and add the new merged one.
R = cholcov(MM2{k1,k2}{2});
K = sum(log(diag(R)));
K_c2 = [K_c(1:s1-1,:); K_c(s1+1:s2-1,:); K_c(s2+1:end,:); [MM2{k1,k2}, R, K]];
indicies(3) = K_c{s1,1}(1); indicies(4) = K_c{s2,1}(1);
if(numel(K_c{1,1})>1) %Not if 1D!!!
    indicies(5) = K_c{s1,1}(2); indicies(6) = K_c{s2,1}(2);
end
%fprintf(1,'Fusing Kernels %d and %d\n',s1,s2); 
%hold on; plot([K_c{s1,1}(1),K_c{s2,1}(1)],[K_c{s1,1}(2),K_c{s2,1}(2)],'g-','linewidth',3);

K_c = K_c2;
% Update the cost list.
% CM(1:k1-1,1:k2-1) = CM2(1:k1-1,1:k2-1);
% CM(1:k1-1,k2:end) = CM2(1:k1-1,k2+1:end); 
% CM(k1:end,1:k2-1) = CM2(k1+1:end,1:k2-1);
% CM(k1:end,k2:end) = CM2(k1+1:end,k2+1:end);

% Update the merge and cost matricies to remove the deleted kernels.
CM = zeros(size(CM2)-1);
MM = cell(size(CM2)-1);
CM2 = removeElementMat(CM2,s1,s1);
CM(1:end-1,1:end-1) = removeElementMat(CM2,s2-1,s2-1);
MM2 = removeElementCell(MM2,s1,s1);
MM(1:end-1,1:end-1) = removeElementCell(MM2,s2-1,s2-1);

% Add back in new merged kernel to the cost and merge matricies.
for k1 = 1 : Nk_max-1
    for k2 = Nk_max
        MM{k1,k2} = util_GaussMerge(K_c(k1,:),K_c(k2,:));
        CM(k1,k2) = K_c{k1,3}*util_GaussKLD(K_c(k1,:),MM{k1,k2}) + K_c{k2,3}*util_GaussKLD(K_c(k2,:),MM{k1,k2});
    end
end
%MM{Nk_max,1} = [];
%MM{Nk_max,1} = [];
% Update the merge list.
% MM{1:k1-1,1:k2-1} = MM2{1:k1-1,1:k2-1};
% MM{1:k1-1,k2:end} = MM2{1:k1-1,k2+1:end}; 
% MM{k1:end,1:k2-1} = MM2{k1+1:end,1:k2-1};
% MM{k1:end,k2:end} = MM2{k1+1:end,k2+1:end};

function M = removeElementMat(M2,k1,k2)
%% function M = removeElementMat(M2,k1,k2)
% Utility function removes row k1 and column k2 from a matrix.
% The new dimension will be one less on each axis.
M = zeros(size(M2)-1);
M(1:k1-1,1:k2-1) = M2(1:k1-1,1:k2-1);
M(1:k1-1,k2:end) = M2(1:k1-1,k2+1:end); 
M(k1:end,1:k2-1) = M2(k1+1:end,1:k2-1);
M(k1:end,k2:end) = M2(k1+1:end,k2+1:end);

function M = removeElementCell(M2,k1,k2)
M = cell(size(M2)-1);
M(1:k1-1,1:k2-1) = M2(1:k1-1,1:k2-1);
M(1:k1-1,k2:end) = M2(1:k1-1,k2+1:end); 
M(k1:end,1:k2-1) = M2(k1+1:end,1:k2-1);
M(k1:end,k2:end) = M2(k1+1:end,k2+1:end);

