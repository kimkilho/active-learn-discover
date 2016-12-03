%% Illustrate KDE library with synthetic data active learning & classification task.

%Make some synthetic data.
[data,labels,R]=util_makeTrainData2;
[data2,labels2]=util_makeTrainData2;
figure(1); clf;
subplot(2,2,1); hold on;
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
subplot(2,2,2); semilogx(lsig,llh); title('LOO likelihood'); xlabel('\sigma'); ylabel('Likelihood'); hold on;
plot(lsig(lsig==sig), llh(lsig==sig),'ro'); axis tight;
fprintf(1,'Choosing sigma: %1.2f\n', sig);

%Show inference for whole space.
maxKernels = 25;
N = size(data,1);
obs = false(N,1);
obs(1) = true;
obs(201) = true;
KDE  = util_createKDE6(data(obs,:),labels(obs),'full',sig,maxKernels);
[pX_Y,Y_X,pY_X,KDE] = util_inferKDE6(data,KDE);
nIter = 50;
tracc = zeros(1,nIter);
teacc = zeros(1,nIter);
for iter = 1 : nIter
    %% Query
    [~,idx] = min(abs(pY_X(:,1)-0.5)); 
    fprintf(1,'Querying pt %d, class %d. ', idx, labels(idx));

    %% Update.
    obs(idx) = true;
    KDE = util_updateKDE6(data(idx,:), labels(idx), KDE);
    
    %% Infer again & visualize.
    [~,Y_X,~,KDE] = util_inferKDE6(data2,KDE,0,0,0,0,0);
    teacc(iter) = sum(Y_X==labels2)/numel(labels2);
    fprintf(1,'Test data accuracy: %1.2f.\n', teacc(iter));
    
    [pX_Y,Y_X,pY_X] = util_inferKDE6(lx,KDE,0,0,0,0,0);
    figure(2); clf; 
    subplot(2,2,1);
    imagesc(ll,ll,reshape(pX_Y(:,1),nll,nll)); title('Likelihood p(X|Y=1)');
    subplot(2,2,2);
    imagesc(ll,ll,reshape(pX_Y(:,2),nll,nll)); title('Likelihood p(X|Y=2)');
    subplot(2,2,3); 
    imagesc(ll,ll,reshape(Y_X,nll,nll)); title('Estimated Y|X');
    subplot(2,2,4); 
    imagesc(ll,ll,reshape(pY_X(:,1),nll,nll)); title('Posterior p(Y=1|X)');

    [pX_Y,Y_X,pY_X] = util_inferKDE6(data,KDE,0,1,1,1,1);
    tracc(iter) = sum(Y_X==labels)/numel(labels);
    figure(3); clf; plot(1:iter, tracc(1:iter), 1:iter, teacc(1:iter)); 
    title('Active Learning'); xlabel('Iterations'); ylabel('Accuracy');
    legend('Train','Test','location','southeast');
    axis([0,nIter,0,1]);
end

