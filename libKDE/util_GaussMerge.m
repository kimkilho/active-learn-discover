function g = util_GaussMerge(g1, g2)
% Merges two Gaussians into a single Gaussian by moment matching.
% function g = util_GaussMerge(g1, g2)
% g{1} = Mean, g{2} = Cov, g{3} = Prior.
% Implements Hospedales PAKDD'11 eq(4)-eq(5).
g = cell(1,3);
g{3} = g1{3} + g2{3};
g{1} = (g1{3}./g{3})*g1{1} + (g2{3}./g{3})*g2{1};
g{2} = (g1{3}./g{3})*( g1{2} + ( g1{1} - g{1} )'*(g1{1} - g{1}) ) ...
     + (g2{3}./g{3})*( g2{2} + ( g2{1} - g{1} )'*(g2{1} - g{1}) );