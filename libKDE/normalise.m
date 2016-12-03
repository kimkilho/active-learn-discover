function [M, z] = normalise(A, dim)
% NORMALISE Make the entries of a (multidimensional) array sum to 1
% [M, c] = normalise(A)
% c is the normalizing constant
%
% [M, c] = normalise(A, dim)
% If dim is specified, we normalise the specified dimension only,
% otherwise we normalise the whole array.
% History:
%   From lightspeed toolbox, speedup added by tmh 2011.

%% Normalize whole array.
if nargin < 2   
  z = sum(A(:));
  % Set any zeros to one before dividing
  % This is valid, since c=0 => all i. A(i)=0 => the answer should be 0/1=0
  s = z + (z==0);
  M = A / s;
%% Normalize each column.
elseif dim==1
  z = sum(A,1);   %Column sums.
  s = z + (z==0); %Avoid divide by zeros.
  M = A ./ (s'*ones(1,size(A,1)))';
  %M = A ./ repmatC(s, size(A,1), 1);
elseif dim==2
  %Added TMH.
  z = sum(A,2);   %Row sums.
  s = z + (z==0); %No div zero.
  M = A ./ (s*ones(1,size(A,2)));
%% Normalize arbitrary dimension.  
else
  % Keith Battocchi - v. slow because of repmat
  z=sum(A,dim);
  s = z + (z==0);
  L=size(A,dim);
  d=length(size(A));
  v=ones(d,1);
  v(dim)=L;
  %c=repmat(s,v);
  c=repmat(s,v');
  M=A./c;
end

