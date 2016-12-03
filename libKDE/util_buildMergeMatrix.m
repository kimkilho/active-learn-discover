function [C,M] = util_buildMergeMatrix(K)
% Build a merge M & cost matrix C from scratch for the kernel structure K.
% function [C,M] = util_buildMergeMatrix(K)
Nk = size(K,1);
C = zeros(Nk,Nk);
M = cell(Nk,Nk);
for k1 = 1 : Nk
    for k2 = k1+1 : Nk
        M{k1,k2} = util_GaussMerge(K(k1,:),K(k2,:));
        kld1 = util_GaussKLD(K(k1,:),M{k1,k2});
        kld2 = util_GaussKLD(K(k2,:),M{k1,k2});
        C(k1,k2) = K{k1,3}*kld1 + K{k2,3}*kld2;
    end
end