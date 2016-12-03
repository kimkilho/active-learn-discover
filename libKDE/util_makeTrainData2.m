function [data,labels,R]=util_makeTrainData2(sig)
%% function [data,labels,observed,R]=makeTrainData2(O)
% Make a spiral dataset.
% Input:
%   Sig: Noise level in data (optional).
% Output:
%	data:  Actual 2d dataset.
%	labels: Labels for each point.
%	R: Region of space over which the dataset covers. For later plotting utils.
if(nargin==0)
    sig=0.05;
end
Neg_c = 200;    %Num points / class.
Neg = 400;      %Total num points.
N_c = 2;        %Num class.
data = zeros(Neg,N_c);
labels = zeros(Neg,1);

figure(1); clf; set(gcf,'Name','Training Data'); hold on;
R = 0.25;       %Radius.
a = 0; b = 0;   %Center.
%sig=0.05;       %Amount of noise.
idx = 1;
d = 4*pi/Neg_c; %Increment.
o=15;           %Offset to avoid a very complicated centre region.
%Difficulty depends alot on how far we start into the center.
for t = o*d:d:(Neg_c+(o-1))*d
    x = a+R*t*cos(t); y = b+R*t*sin(t);
    xn = x+randn*sig; yn=y+randn*sig;
    plot(xn, yn, 'kx','markersize',10);
    
    data(idx,:) = [xn,yn];
    labels(idx) = 1;
    idx = idx+1;
end
a = 0; b = 0.5/4;
%for t = o*d:d:(150+(o-1))*d
for t = o*d:d:(Neg_c+(o-1))*d
    x = a-R*t*cos(t); y = b-R*t*sin(t);  
    xn = x+randn*sig; yn=y+randn*sig;
    plot(xn, yn, 'bo','markersize',10);
    
    data(idx,:) = [xn,yn];
    labels(idx) = 2;
    idx = idx+1;
end

R = [-4,4,-4,4,0.1];