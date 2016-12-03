function alpha_est = util_est_alpha(alpha_init, k, n, a, b)
% Estimate concentration parameter for dirichlet process.
% function alpha_est = util_est_alpha(alpha_init, k, n)
% Input:
%   k: Number of classes discovered.
%   n: Number of samples.
%   alpha_init: Initial value to use for alpha. (doesn't matter?)
%   a,b: Gamma prior (optional, default 1,1).
% Output:
%   alpha_est: Mean of \alpha posterior. Computed by Gibbs sampling method from Escobar & West'95.

%p(\alpha) = G(a,b)
if(nargin<4)
    a = 1; 
    b = 1;
end

assert(n>=1); %At least one point to make sense of estimation.
assert(k>=1); %At least one class to make sense.
assert(n>=k); %K can't be greater than n.

alpha = alpha_init;
Ns = 100;
a_log = zeros(1,Ns);

for i = 1 : Ns
    %eta | a,k = B(a+1,n)
    eta = betarnd(alpha+1, n);
    %ratio = (a + k - 1)/(n*(b-log(eta)))
    p1 = (a + k - 1);
    p2 = (n*(b-log(eta)));
    nc = (p1+p2);
    p1 = p1/nc;
    %p2 = p2/nc;
    %p1/p2

    mixcomp = binornd(1,p1);
    if(mixcomp==1)
        alpha = gamrnd(a+k,b-log(eta));
        %alpha = gamrnd(a+k,1./(b-log(eta)));
    else
        alpha = gamrnd(a+k-1,b-log(eta));
        %alpha = gamrnd(a+k-1,1./(b-log(eta)));
    end
    a_log(i) = alpha;
end

%figure(1); clf; plot(a_log);

alpha_est = mean(a_log(end/2:end));