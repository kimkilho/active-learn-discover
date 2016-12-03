%% Illustrate KDE library with synthetic data classification task.

%Make some synthetic data.
[data,labels,R]=util_makeTrainData2;
[data2,labels2]=util_makeTrainData2;
figure(1); clf;
subplot(2,3,1); hold on;
plot(data(labels==1,1), data(labels==1,2),'bo'); 
plot(data(labels==2,1), data(labels==2,2),'rx'); 
title('True Dist');

ll = R(1):0.05:R(2); nll = numel(ll);
[lj,li]=meshgrid(ll,ll);
lx = [lj(:),li(:)];

%Optimize \sigma on unconditional full dataset.
KDEtmp = util_createKDE6(data);
[sig,llh,lsig] = util_opt_sigma_loo_KDE2(KDEtmp.K_c{1,1},1);

%Results.
subplot(2,3,2); semilogx(lsig,llh); title('LOO likelihood'); xlabel('\sigma'); ylabel('Likelihood'); hold on;
plot(lsig(lsig==sig), llh(lsig==sig),'ro'); axis tight;
fprintf(1,'Choosing sigma: %1.2f\n', sig);

%Show inference for whole space.
maxKernels = 25;
KDE  = util_createKDE6(data,labels,'full',sig,maxKernels);
[pX_Y,Y_X,pY_X] = util_inferKDE5(lx,KDE);
subplot(2,3,3);
imagesc(ll,ll,reshape(pX_Y(:,1),nll,nll)); title('Likelihood p(X|Y=1)');
subplot(2,3,4);
imagesc(ll,ll,reshape(pX_Y(:,2),nll,nll)); title('Likelihood p(X|Y=2)');
subplot(2,3,5); 
imagesc(ll,ll,reshape(Y_X,nll,nll)); title('Estimated Y|X');
subplot(2,3,6); 
imagesc(ll,ll,reshape(pY_X(:,1),nll,nll)); title('Posterior p(Y=1|X)');

[~,Y_X] = util_inferKDE6(data2,KDE);
fprintf(1,'Test data accuracy: %1.2f\n', sum(Y_X==labels2)/numel(labels2));