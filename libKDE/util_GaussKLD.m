function d = util_GaussKLD(g1,g2)
% Returns the KLD between two Gaussians. 
% function d = util_GaussKLD(g1,g2)
% Implements Hospedales PAKDD'11 eq (6).

Nx=numel(g1{1});
%d = 0.5*( log(det(g2{2})/det(g1{2})) + trace(inv(g2{2})*g1{2}) + (g2{1}-g1{1})*inv(g2{2})*(g2{1}-g1{1})' - Nx);
d = 0.5*( log(det(g2{2})/det(g1{2})) + trace(g2{2}\g1{2}) + (g2{1}-g1{1})*(g2{2}\(g2{1}-g1{1})') - Nx);
