Data=importdata('yeast.data');
Nd = size(Data,1);
Nx = 8;
D  = zeros(Nd,Nx);
Yc = cell(Nd,1);
Y  = zeros(Nd,1);
for i = 1 : Nd
    a=textscan(Data{i},'%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%s');
    D(i,:) = [a{2:end-1}];
    Yc{i}  = a{end}{1};
end
clist = unique(Yc);
%Setup class labels.
Ny = numel(clist);
for i = 1 : Nd
    c = find(strcmp(clist,Yc{i}));
    Y(i) = c;
end
%
h = hist(Y,[1:Ny])
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

P.Ny = Ny;
for f = 1 : Ncv
    teidx = Nte*(f-1)+1:Nte*f;
    tridx = setdiff(1:Nd,teidx);    
    CV(f).Xtr = X( randidx(tridx), :);
    CV(f).Xte = X( randidx(teidx), :);
    CV(f).Ytr = Y( randidx(tridx), :); 
    CV(f).Yte = Y( randidx(teidx), :);
    h=hist(CV(f).Ytr,[1:P.Ny])
    h=hist(CV(f).Yte,[1:P.Ny])
end
P.Ncv = 2;
P.Ntr = Ntr;
P.Nte = Nte;
P.Nx  = Nx;
P.Ny = Ny;
save('yeast.mat','X','Y','CV','Ntr','Nte','Ncv','P');
