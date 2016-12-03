D=importdata('glass.data'); %From UCI.
Nd = size(D,1);
Nx = 10;
Y  = D(:,end);
D  = D(:,1:end-1);
Ny  = 6;
Y(Y>3) = Y(Y>3)-1; %Fix class IDs.
h = hist(Y,1:Ny)
mean(D,1)
std(D,1)
%Class is age.
X = D-repmat(mean(D,1),[Nd,1]);
X = X./repmat(std(X,[],1),[Nd,1]);
mean(X,1)
std(X,[],1)
Ncv = 2;
Ntr = ceil(Nd/Ncv);
Nte = Nd-Ntr;
randidx = randperm(Nd);
for f = 1 : Ncv
    teidx = Nte*(f-1)+1:Nte*f;
    tridx = setdiff(1:Nd,teidx);    
    CV(f).Xtr =  X( randidx(tridx), :);
    CV(f).Xte =  X( randidx(teidx), :);
    CV(f).Ytr = Y( randidx(tridx), :); 
    CV(f).Yte = Y( randidx(teidx), :);    
    h=hist(CV(f).Ytr,1:Ny)
end
P.Ny  = Ny;
P.Ncv = 2;
P.Ntr = Ntr;
P.Nte = Nte;
P.Nx  = Nx;
save('glass.mat','X','Y','CV','Ntr','Nte','Ncv','P');