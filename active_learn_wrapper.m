function active_learn_wrapper(DATASET,CONT,nAiter,domodels,fold)
%% function active_learn_wrapper(DATASET,CONT,nAiter,domodels,fold)
% DATASET:  Load this mat file with input data.
% CONT:     0=Optimize KDE bandwidth params, 1=Guess KDE params.
% nAiter:   Number of active learning iterations. Default: min(150, numTrain)
% domodels: Which models to run? Default 1:6.
% fold:     Which cross-validation fold to take in the input data structure.

code = '/g_npb3/';      %Output directory to use.
addpath('./libMisc/');
addpath('./libsvm/');
addpath('./libDPEA/');
addpath('./libKDE/');

global Nk_max covType VERBOSE;  %Max complexity, KDE Gaussian covariance.
Nk_max  = inf; 
VERBOSE = -1;
covType = 2;
str = ['data/',DATASET,'.mat'];
if(VERBOSE>=0), fprintf(1,'Loading %s.\n',str);  end
D=load(str); 
CV = D.CV; 
P  = D.P;
clear('D');
fprintf(1,'Stats: Ny = %d, Nx = %d, Ntr = %d, Nte = %d.\n', P.Ny, P.Nx, P.Ntr, P.Nte);

P.Nk_max = Nk_max;
P.Nx = size(CV(1).Xtr,2);
if(nargin<5),
    fold = 1;
end
CV = CV(fold);
try
    Ntr = CV.Ntr;
    P.Ntr = CV.Ntr;
    P.Nte = CV.Nte;
catch
    Ntr = P.Ntr;
    %disp('Failed to extract # samples from CV struct. Assuming its in P struct.');
end
P.DATASET = DATASET;
if(nargin==0)
    CONT=1;
end
if(CONT==0) %Do initial reoptimzation
    %% Optimize KDE parameters.
    for f = 1
        idx=randperm(Ntr); Ntr_max = 1600;
        idx=idx(1:min(numel(idx),Ntr_max));
        obslist = false(Ntr,1); obslist(idx)=1;
        KDE = util_createKDE6(CV.Xtr(obslist,:),ones(sum(obslist),1),'ud',1); %[KDE] = util_createKDE6(data,labels,covType,sigma,Nk_max,Nc)
        [S_c,llh2,lsig] = util_opt_sigma_loo_KDE2(KDE.K_c{1,1},1);
        fprintf(1,'*** Selected S_c = %1.3f.\n', S_c);
    end   
else
    S_c = sqrt(P.Nx);
end
P.CONT = CONT;
P.S_c  = S_c;
P.fold = f;

if(nargin<3)
    nAiter = min(150,Ntr-1);
end
nAiter = min(nAiter,Ntr-1);

%% Shared AL Initilization.
%Nk_max = inf;
Nk_max = 32;
P.Nk_max = Nk_max;
P.Nobs_init1 = 1; %Observe first point.
P.KDE_prior = true;
al_init.obs_init = false(Ntr,1);
l11 = find(CV.Ytr==1); l11=l11(randperm(numel(l11)));
al_init.obs_init(l11(1:P.Nobs_init1)) = true;
al_init.idx = al_init.obs_init;    %Because we chose class 1 to start.
al_init.obs_bits= al_init.obs_init;
al_init.obs_idx = find(al_init.obs_bits);
al_init.idx_log = al_init.obs_idx';
al_init.tag = [];
al_init.corr_log = 1;
al_init.pidx = zeros(size(CV.Ytr));
al_init.uniq_log = [];
al_init.newClass = false;
al_init.seenC  = unique(CV.Ytr(al_init.obs_bits));
al_init.nSeenC = numel(al_init.seenC);
al_init.VERBOSE = 0;
al_init.hasFusion = 0;
al_init.hasKDE = 0;
al_init.hasSVM = 0;
al_init.R.belif_log = [];
al_init.variant = '';
al_init.CM = 0;
al_init.descr = [];
al_init.svmtrainstr = '';
al_init.lastClass = 1;
P.svmtrainstr = '-q -b 1';
P.svmtrainstr = sprintf('-b 1 -q -c 1 ');
%P.svmtrainstr = sprintf('-b 1 -c 1 ');
P.svmteststr = '';
P.gpinitlh = [0; 0];
P.fusion_gmax = 1;
P.fusion_beta = 100;
P.fusion_alpha = 0.1;

%% Pre-compute density.
np2 = 0; j=1;
while(any(np2==0) && j < 10) %Try ten times and then just cap    
    Ntr_max = 2500;
    fprintf(1,'Bulding prior density (1)...\n');
    idx=randperm(Ntr); 
    idx=idx(1:min(Ntr,Ntr_max));
    obslist = false(Ntr,1);  obslist(idx)=1;
    
    KDE = util_createKDE6(CV.Xtr(obslist,:),ones(sum(obslist),1),'full',S_c);
    np2 = util_inferKDE6(CV.Xtr,KDE);
    j=j+1;    
end
if(any(np2==0))
    system(['echo 1 >> ', 'results/',DATASET,code,'cap.txt']);
    np2(np2==0)=min(np2(np2>0));
end

%% Assume all models.
if(nargin<4)
    domodels = [1:6];
end

%% Init models to compare
if(any(domodels==1))
    AL(1) = initAL__SVM_Random(al_init,P,CV);
end
if(any(domodels==2))
    AL(2) = initAL__SVM_Marg(al_init,P,CV);
end
if(any(domodels==3))
    AL(3) = initAL_KDE__Lik(al_init, P, CV, S_c);
end
if(any(domodels==4))
    AL(4) = initAL_KDE_pWrong(al_init, P, CV,S_c,np2, 1, 1);
end
if(any(domodels==5))
    AL(5) = initAL_KDE_SVM_Fusion5(al_init,P,CV,S_c); 
end
if(any(domodels==6))
    AL(6) = initAL_KDE_NPB(al_init, P, CV, S_c,  np2,  0,   1,  0,  1,  0, 128, 'pWrong'); %Expected to be the main one.
end


nAL = numel(AL);
for m = 1 : nAL
    AL(m).variant =  [AL(m).variant, num2str(m)];
end

%% Setup results structure
P.nAiter=nAiter;
P.tagcell = {AL.tag};
for m = domodels
    AL(m).fname = ['results/',DATASET,code,AL(m).tag,AL(m).variant,'.mat']; 
    R.P.starttime = datestr(now);
    if(~exist(AL(m).fname,'file'))       
        fprintf(1,'%s doesnt exist, creating..\n',AL(m).fname);
        R.debug.siglog = [];
        R.tr.ASRLOG  = zeros(0, nAiter); 
        R.tr.BSRLOG  = zeros(0, nAiter); 
        R.te.ASRLOG  = zeros(0, nAiter); 
        R.te.BSRLOG  = zeros(0, nAiter); 
        R.posts.seen = cell(0,0);
        R.IDXLOG     = zeros(0, nAiter+numel(AL(m).idx_log)); %One per iteration + initialized obs.
        if(AL(m).hasFusion)
            if(AL(m).hasSVM && AL(m).hasKDE)
                R.posts.histKDE = zeros(P.Ny,nAiter,0);
                R.posts.histSVM = zeros(P.Ny,nAiter,0);
                R.posts.entKDE = zeros(0, nAiter);
                R.posts.entSVM = zeros(0, nAiter);
                %Debug things.
                R.debug.histOBS = zeros(P.Ny, nAiter, 0);  
                R.tr.kASRLOG  = zeros(0, nAiter); 
                R.tr.sASRLOG  = zeros(0, nAiter); 
                R.te.kASRLOG  = zeros(0, nAiter); 
                R.te.sASRLOG  = zeros(0, nAiter);            
            end
            R.BLLOG  = zeros(0, nAiter); 
            R.FWLOG  = zeros(nAiter, AL(m).M.k_alg, 0);
        end;
        R.UCLOG  = zeros(0, nAiter);
        R.CMLOG  = zeros(P.Ny,P.Ny,0);
        save(AL(m).fname,'R','P');
    end
end

%% AL Iterations.
for a = 1 : nAiter    
    P.a = a;
    tic;
    %% Query a point.
    for m = domodels
        AL(m) = queryGenericAL(AL(m), P, CV);
    end
    %% Learn from given query.
    for m = domodels
        AL(m) = learnGenericAL(AL(m), P, CV);        
    end
    %keyboard;
    %% Infer new estimates.
    for m = domodels
        AL(m) = inferGenericAL(AL(m), P, CV);
    end
    
    t=toc;
    j=1;
    for m = domodels
        choices(j) = numel(unique(AL(m).tr.labels));
        acc(j) = AL(m).te.am_all_srlog(end);
        j=j+1;
    end
    fprintf(1,'--- V3b:%s%s: %d/%d iteration time: %d. ETA: %d min. unique preds: %s, acc: %s.---\n', DATASET, code, a, nAiter, round(t),round((nAiter-a)*t/60),num2str(choices), num2str(acc, '%0.2f  '));
end

%% Saving results.
for m = domodels
    load(AL(m).fname);
    al = AL(m);
    al = rmfield(al,'tr');
    al = rmfield(al,'te');
    al = rmfield(al,'R');
    %al.M.KDE is the resulting KDE model. 
    %The following line removes the merge matrix to save space. Comment out if you want to be able to continue to learn it incrementally.
    if(AL(m).hasKDE && isfield(al.M.KDE,'MM')), al.M.KDE = rmfield(al.M.KDE,'MM'); end
    R.al = al;
    R.P = P;
    R.debug.siglog = [R.debug.siglog, P.S_c];
    R.P.endtime = datestr(now);
    try R.IDXLOG = [R.IDXLOG; AL(m).idx_log]; catch, end
    R.tr.ASRLOG = [R.tr.ASRLOG; AL(m).tr.am_all_srlog];
    R.tr.BSRLOG = [R.tr.BSRLOG; AL(m).tr.anomal_srlog];
    R.te.ASRLOG = [R.te.ASRLOG; AL(m).te.am_all_srlog];
    R.te.BSRLOG = [R.te.BSRLOG; AL(m).te.anomal_srlog];
    R.UCLOG = [R.UCLOG; AL(m).uniq_log];
    R.CMLOG = cat(3, R.CMLOG, AL(m).CM);
    R.posts.seen{end+1} = AL(m).seenC;
    
    if(AL(m).hasFusion)
        R.FWLOG = cat(3, R.FWLOG, AL(m).R.fwlog);
        R.BLLOG = [R.BLLOG; AL(m).R.belif_log];
        if(AL(m).hasKDE&&AL(m).hasSVM)
            R.posts.histKDE = cat(3, R.posts.histKDE, AL(m).R.histKDE); 
            R.posts.histSVM = cat(3, R.posts.histSVM, AL(m).R.histSVM);
            R.posts.entKDE = [R.posts.entKDE; AL(m).R.entKDE]; 
            R.posts.entSVM = [R.posts.entSVM; AL(m).R.entSVM];
            
            %Debug things.
            R.debug.histOBS = cat(3, R.debug.histOBS, AL(m).R.histOBS);
            R.tr.kASRLOG  = [R.tr.kASRLOG; AL(m).R.tr_ksrlog];
            R.tr.sASRLOG  = [R.tr.sASRLOG; AL(m).R.tr_ssrlog];
            R.te.kASRLOG  = [R.te.kASRLOG; AL(m).R.te_ksrlog];
            R.te.sASRLOG  = [R.te.sASRLOG; AL(m).R.te_ssrlog];
        end
    end
    save(AL(m).fname,'R','P');
end
P.tagcell = {AL.tag};


function AL = inferGenericAL(AL, P, CV)
global VERBOSE
elected=false;
a = P.a;
if(strcmp(AL.tag,'__SVM_Rand') || strcmp(AL.tag,'_SVM_Marg'))
    if(AL.nSeenC>1)
        [tmp, tmp, AL.tr.post] = svmpredict(CV.Ytr, CV.Xtr, AL.M.libsvmmodel, ['-b 1', P.svmteststr]);
    else
        AL.tr.post = ones(P.Ntr,1);
    end    
    
    
    if(AL.nSeenC>1)
        [AL.tr.labels, accuracy, tmp] = svmpredict(CV.Ytr, CV.Xtr, AL.M.libsvmmodel, ['-b 1', P.svmteststr]);
        [AL.te.labels, accuracy, AL.te.post] = svmpredict(CV.Yte, CV.Xte, AL.M.libsvmmodel, ['-b 1', P.svmteststr]);
    else
        AL.tr.labels(:) = AL.lastClass;
        AL.te.labels(:) = AL.lastClass;
        accuracy = 1/P.Ny;
    end
        
    AL.te.srlog(a)=accuracy(1);
    elected=true;

    seenhist = hist(CV.Ytr(AL.obs_bits), 1:P.Ny);
    if(VERBOSE>=0)
    fprintf('%s Update: Seen: %d. (%s) \n', ...
        AL.tag, AL.nSeenC, int2str(seenhist));    
    end
end

if(strcmp(AL.tag, 'KDE__NPB') || strcmp(AL.tag,'KDE__pWrong'))
    [~, AL.tr.labels, ~, AL.M.KDE] = util_inferKDE6(CV.Xtr,AL.M.KDE,AL.M.KDE_prior,1,1,1,1);
    [AL.te.knp2, AL.te.labels, AL.te.post] = util_inferKDE5(CV.Xte,AL.M.KDE,AL.M.KDE_prior,AL.te.knp2,AL.lastClass);
    
    AL.tr.srlog(a) = 100*sum(AL.tr.labels==CV.Ytr)/numel(CV.Ytr);    
    AL.te.srlog(a) = 100*sum(AL.te.klabels==CV.Yte)/numel(CV.Yte);    
    elected=true;
    if(AL.VERBOSE)
        seenhist = hist(CV.Ytr(AL.obs_bits), 1:P.Ny);
        fprintf('%s: Update: Seen: %d. (%s) \n', ...
            AL.tag, AL.nSeenC, int2str(seenhist));    
    end    
    if(AL.M.adaptAlpha)
        AL.M.dpalpha = util_est_alpha(AL.M.dpalpha, AL.nSeenC, a+1);
    end

end
if(strcmp(AL.tag,'KDE__Lik'))
%% This block for exclusively KDE models. No need for slabels, spost, etc.   
    [~, AL.tr.labels, ~, AL.M.KDE] = util_inferKDE6(CV.Xtr,AL.M.KDE,AL.M.KDE_prior,1,1,1,1);
    [AL.te.knp2, AL.te.labels, AL.te.post] = util_inferKDE5(CV.Xte,AL.M.KDE,AL.M.KDE_prior,AL.te.knp2,AL.lastClass);
    
    %[AL.tr.labels, AL.tr.knp2, AL.tr.post] = util_inferKDE2(CV.Xtr,AL.M.K_c,AL.M.KDE_prior,AL.tr.knp2,AL.lastClass);
    AL.tr.srlog(a) = 100*sum(AL.tr.labels==CV.Ytr)/numel(CV.Ytr);
    %[AL.te.labels, AL.te.knp2, AL.te.post] = util_inferKDE2(CV.Xte,AL.M.K_c,AL.M.KDE_prior,AL.te.knp2,AL.lastClass);
    AL.te.srlog(a) = 100*sum(AL.te.klabels==CV.Yte)/numel(CV.Yte);    
    elected=true;
    if(AL.VERBOSE)
        seenhist = hist(CV.Ytr(AL.obs_bits), 1:P.Ny);
        fprintf('%s: Update: Seen: %d. (%s) \n', ...
            AL.tag, AL.nSeenC, int2str(seenhist));    
    end
end
if(strcmp(AL.tag,'KDE_SVM_Fusion5'))
    AL.M.Hold = AL.M.H;
    %Predict with KDE&SVM.
    [AL.tr.knp2, AL.tr.klabels, AL.tr.kpost, AL.M.KDE] = util_inferKDE6(CV.Xtr,AL.M.KDE,AL.M.KDE_prior,1,1,1,1);   
    %[AL.tr.klabels,AL.tr.knp2,AL.tr.kpost] = util_inferKDE2(CV.Xtr,AL.M.K_c,AL.M.KDE_prior,AL.tr.knp2,AL.lastClass);
    
    [AL.tr.slabels, accuracy, tmp] = svmpredict(CV.Ytr, CV.Xtr, AL.M.libsvmmodel, [P.svmteststr]);
    if(AL.nSeenC>1)
        [tmp, tmp, AL.tr.spost] = svmpredict(CV.Ytr, CV.Xtr, AL.M.libsvmmodel, ['-b 1', P.svmteststr]);
    else
        AL.tr.spost = ones(P.Ntr,1);
    end
    %Calculate relative entropies.
    AL.tr.kh = hist(AL.tr.klabels,1:P.Ny);
    AL.tr.sh = hist(AL.tr.slabels,1:P.Ny);
    kh = AL.tr.kh(AL.tr.kh>0); kh=kh/sum(kh); AL.tr.ke = -sum(kh.*logb(kh,AL.nSeenC));% AL.tr.ke = -sum(kh.*log(kh)); %
    sh = AL.tr.sh(AL.tr.sh>0); sh=sh/sum(sh); AL.tr.se = -sum(sh.*logb(sh,AL.nSeenC));%AL.tr.se = -sum(sh.*log(sh)); %
    [AL.te.slabels, accuracy, AL.te.spost] = svmpredict(CV.Yte, CV.Xte, AL.M.libsvmmodel, P.svmteststr);
    %[AL.te.klabels,AL.te.knp2,AL.te.kpost] = util_inferKDE2(CV.Xte, AL.M.K_c, AL.M.KDE_prior, AL.te.knp2, AL.lastClass);
    [AL.te.knp2, AL.te.klabels, AL.te.kpost] = util_inferKDE5(CV.Xte,AL.M.KDE,AL.M.KDE_prior,AL.te.knp2,AL.lastClass);
    if(AL.tr.se>AL.tr.ke) %SVM Entropy more. Then svm predict for test.
        AL.tr.labels = AL.tr.slabels;
        AL.tr.post   = AL.tr.spost;
        AL.te.labels = AL.te.slabels;
        AL.te.post   = AL.te.spost;
    else %KDE entropy more. Then KDE predict for test.
        AL.tr.labels = AL.tr.klabels;
        AL.te.labels = AL.te.klabels;
        if(AL.M.switchPost)
            AL.tr.post   = AL.tr.kpost;
            AL.te.post   = AL.te.kpost;
        else
            AL.tr.post   = AL.tr.spost;
            AL.te.post   = AL.te.spost;
        end
    end
    elected=true;
        
    %Entropy
    lh = hist(AL.tr.labels,AL.seenC); %Hist of observed classes only.
    cperc = lh/P.Ntr;       %Inferred classes as %
    cperc = normalise(cperc+1/P.Ntr); %Avoid zero classes! :-(.
    if(AL.nSeenC==1)
        AL.M.H=0;
    else
        AL.M.H = -sum(cperc.*logb(cperc,AL.nSeenC)); %Entropy
    end
    
    AL.M.R = AL.newClass + max(0,((exp(AL.M.H)-exp(AL.M.Hold))-(1-exp(1))/(2*exp(1)-2)));
        
    AL.M.Rh = AL.M.R*AL.M.bidx(AL.idx,:)'/AL.pidx(AL.idx); %Expert's reward proportional to how much he liked it. (y^hat_i in Auer02)
    %Weight update
    AL.M.wold = AL.M.w;  
    AL.M.Rhn = normalise(AL.M.Rh);
    AL.M.w = normalise(0.6*AL.M.wold +0.375*AL.M.Rhn'+0.025);%.* exp( AL.M.Rh' / AL.M.neffpts );
    
    %Gain log.
    AL.R.fwlog = [AL.R.fwlog; AL.M.w];    
    AL.R.ghlog = [AL.R.ghlog; AL.M.Rh'];
    
    seenhist = hist(CV.Ytr(AL.obs_bits), 1:P.Ny);
    if(VERBOSE>=0)
    msg{1}='-'; msg{2}='+';
    fprintf('5: Fusion Update: H: %1.2f->%1.2f: %s. R: (%1.2f,%1.2f). Weights: [%1.2f,%1.2f]. Seen: %d. (%s) \n', ...
        AL.M.Hold, AL.M.H,msg{(1+((AL.M.H-AL.M.Hold)>0) )}, AL.M.Rh, AL.M.w, AL.nSeenC, int2str(seenhist));
    end
end

if(AL.hasFusion && AL.hasSVM && AL.hasKDE)
    AL.R.histKDE(:,P.a) = hist(AL.tr.klabels,[1:P.Ny]);
    AL.R.histSVM(:,P.a) = hist(AL.tr.slabels,[1:P.Ny]);
    kh = AL.R.histKDE(AL.R.histKDE(:,P.a)>0,P.a); kh=kh/sum(kh); 
    sh = AL.R.histSVM(AL.R.histSVM(:,P.a)>0,P.a); sh=sh/sum(sh); 
    AL.R.entKDE(a) = -sum(kh.*log(kh));
    AL.R.entSVM(a) = -sum(sh.*log(sh));
    
    AL.R.te.histKDE(:,P.a) = hist(AL.te.klabels,[1:P.Ny]);
    AL.R.te.histSVM(:,P.a) = hist(AL.te.slabels,[1:P.Ny]);
    kh = AL.R.te.histKDE(AL.R.te.histKDE(:,P.a)>0,P.a); kh=kh/sum(kh); 
    sh = AL.R.te.histSVM(AL.R.te.histSVM(:,P.a)>0,P.a); sh=sh/sum(sh); 
    AL.R.te.entKDE(a) = -sum(kh.*log(kh));
    AL.R.te.entSVM(a) = -sum(sh.*log(sh));    
    
    %Debug
    AL.R.histOBS(:,P.a)    = hist(CV.Ytr(AL.obs_idx),1:P.Ny);
    %Train
    srs = zeros(P.Ny,1); srk = zeros(P.Ny,1);
    for y = 1:P.Ny
        srk(y) = sum((AL.tr.klabels==y)&(CV.Ytr==y))/sum(CV.Ytr==y);
        srs(y) = sum((AL.tr.slabels==y)&(CV.Ytr==y))/sum(CV.Ytr==y);
    end  
    AL.R.tr_ksrlog(a) = mean(srk);     
    AL.R.tr_ssrlog(a) = mean(srs);     
    %Test
    srs = zeros(P.Ny,1); srk = zeros(P.Ny,1);
    for y = 1:P.Ny
        srk(y) = sum((AL.te.klabels==y)&(CV.Yte==y))/sum(CV.Yte==y);
        srs(y) = sum((AL.te.slabels==y)&(CV.Yte==y))/sum(CV.Yte==y);
    end  
    AL.R.te_ksrlog(a) = mean(srk);     
    AL.R.te_ssrlog(a) = mean(srs);   
end

AL.CM = confusion_matrix(P.Ny, CV.Ytr, AL.tr.labels);
AL.te.CM = confusion_matrix(P.Ny, CV.Yte, AL.te.labels);

assert(elected==true);
%Train.
AL.tr.normal_srlog(a) = sum((AL.tr.labels==1)&(CV.Ytr==1))/sum(CV.Ytr==1);


AL.tr.anomal_srlog(a) = sum((AL.tr.labels~=1)&(CV.Ytr~=1))/sum(CV.Ytr~=1);
sr = zeros(P.Ny,1);
for y = 1:P.Ny
    sr(y) = sum((AL.tr.labels==y)&(CV.Ytr==y))/sum(CV.Ytr==y);
end

    AL.tr.am_all_srlog(a) = mean(sr);
AL.tr.gm_all_srlog(a) = sqrt(prod(sr));


%Test
AL.te.normal_srlog(a) = sum((AL.te.labels==1)&(CV.Yte==1))/sum(CV.Yte==1);
AL.te.anomal_srlog(a) = sum((AL.te.labels~=1)&(CV.Yte~=1))/sum(CV.Yte~=1);
sr = zeros(P.Ny,1);
for y = 1:P.Ny
    sr(y) = sum((AL.te.labels==y)&(CV.Yte==y))/sum(CV.Yte==y);
end

AL.te.am_all_srlog(a) = mean(sr);
AL.te.gm_all_srlog(a) = sqrt(prod(sr));



function AL = queryGenericAL(AL, P, CV)
elected=false;
if(strcmp(AL.tag,'__SVM_Rand'))
    [AL.idx, AL.tr.labels, AL.obs_bits, AL.pidx] = ElectPointRandom(AL.tr.labels, AL.obs_bits, CV.Ytr);
    elected=true;
end
if(strcmp(AL.tag,'_SVM_Marg'))
    [AL.idx,AL.tr.labels, AL.obs_bits, AL.pidx] = ElectPointEntropy(AL.tr.labels, AL.obs_bits, CV.Ytr, AL.tr.post, AL.nSeenC);
    elected=true;
end
if(strcmp(AL.tag,'KDE__Lik'))
    [AL.idx, AL.tr.labels, AL.obs_bits, AL.M.eidx] = ElectPointLik(AL.tr.labels, AL.obs_bits, CV.Ytr, AL.M.KDE.likCache);
    elected=true;    
end
if(strcmp(AL.tag,'KDE__pWrong'))
    opts.alpha = AL.M.dpalpha;
    opts.usePrior = 1;
    opts.sampQuery = ~AL.M.greedy;
    assert(CV.Ytr(AL.idx)==AL.lastClass);
    %[AL.idx,AL.M.eidx] = util_evalPwrong(CV.Xtr,AL.M.KDE,AL.obs_bits,opts,AL.M.pX,AL.tr.knp2,AL.lastClass); %V5 KDE.
    [AL.idx,AL.M.eidx,AL.M.KDE] = util_evalPwrong2(CV.Xtr,AL.M.KDE,AL.obs_bits,opts,AL.M.pX); %V6 KDE, internal cache.
    AL.obs_bits(AL.idx) = true;
    elected = true;
end
if(strcmp(AL.tag,'KDE__NPB'))
    %[AL.idx, AL.tr.labels, AL.obs_bits, AL.M.eidx, AL.M.KDE] = ElectPointEEDP(AL, P, CV.Xtr, AL.tr.labels, AL.obs_bits, CV.Ytr);
    [AL.idx, AL.tr.labels, AL.obs_bits, AL.M.eidx, AL.M.KDE] = ElectPointEEDP6(AL, P, CV.Xtr, AL.tr.labels, AL.obs_bits, CV.Ytr);
    elected = true;
end

if(strcmp(AL.tag,'KDE_SVM_Fusion5'))
    k_alg = AL.M.k_alg;
    AL.M.idx  = zeros(1,k_alg);
    AL.M.eidx = zeros(P.Ntr, k_alg);
    AL.M.bidx = zeros(P.Ntr, k_alg);
    
    [AL.M.idx(1), tmp, tmp, AL.M.eidx(:,1)] = ElectPointLik(AL.tr.labels, AL.obs_bits, CV.Ytr, AL.M.KDE.likCache);
    [AL.M.idx(1), tmp, tmp, AL.M.eidx(:,2)] = ElectPointEntropy(AL.tr.labels, AL.obs_bits, CV.Ytr, AL.tr.post, AL.nSeenC);
    
    %beta=500;%
    beta=AL.M.fusion_beta;
    %beta=50;
    AL.M.bidx  = normalise(exp(-beta*(1-AL.M.eidx)),1);
    
    exp_idx = sum(cumsum(AL.M.w)<rand)+1;
    
    AL.M.bidx  = normalise(AL.M.bidx,1);
    
    AL.pidx = normalise(AL.M.bidx(:,exp_idx));
    
    if(AL.M.greedy)
        [tmp, AL.idx] = max(AL.pidx);
    else
        AL.idx = sum(cumsum(AL.pidx) < rand) + 1;
    end
    
    AL.tr.labels(AL.idx) = CV.Ytr(AL.idx);
    AL.obs_bits(AL.idx) = true;
    elected=true;
    AL.R.belif_log = [AL.R.belif_log, exp_idx];
end

AL.obs_idx = find(AL.obs_bits);    
AL.lastClass = CV.Ytr(AL.idx);
assert(elected==true);

%Discovered classes.
AL.newClass = numel(unique(CV.Ytr(AL.obs_bits)))>AL.nSeenC;
AL.seenC  = unique(CV.Ytr(AL.obs_bits));
AL.nSeenC = numel(AL.seenC);
AL.uniq_log  = [AL.uniq_log, numel(AL.seenC)];
AL.idx_log = [AL.idx_log, AL.idx];
%Log the latest selected point in order.
if(P.a>1)
    AL.corr_log = [AL.corr_log, AL.tr.labels(AL.idx)~=AL.tr.oldlabels(AL.idx)];    
end


function AL = learnGenericAL(AL, P, CV)
AL.tr.oldlabels = AL.tr.labels; elected=false;

AL.svmtrainstr = P.svmtrainstr;
if(strcmp(AL.tag,'__SVM_Rand') || strcmp(AL.tag,'_SVM_Marg'))
    AL.M.libsvmmodel = libsvmtrain(CV.Ytr(AL.obs_bits), CV.Xtr(AL.obs_bits,:), P.svmtrainstr);
    elected=true;
end
if(strcmp(AL.tag,'KDE__NPB') || strcmp(AL.tag,'KDE__pWrong'))
    %AL.M.KDE = util_updateKDE5(CV.Xtr(AL.idx,:), CV.Ytr(AL.idx), AL.M.KDE); %Caching, etc. %V5 api.
    AL.M.KDE = util_updateKDE6(CV.Xtr(AL.idx,:), CV.Ytr(AL.idx), AL.M.KDE); %Caching, etc. %V6 api.
    elected=true;
end
if(strcmp(AL.tag,'KDE__Lik'))
    %[AL.M.K_c,AL.M.CM,AL.M.MM,AL.M.IN] = util_updateKDE2b(CV.Xtr,CV.Ytr,AL.idx,AL.M.K_c,AL.M.S_c,AL.M.Nk_max,AL.M.CM, AL.M.MM, AL.M.IN);    
    AL.M.KDE = util_updateKDE6(CV.Xtr(AL.idx,:), CV.Ytr(AL.idx), AL.M.KDE); %Caching, etc. %V6 api.
    
    elected=true;
end

if(strcmp(AL.tag,'KDE_SVM_Fusion5'))
    AL.M.KDE = util_updateKDE6(CV.Xtr(AL.idx,:), CV.Ytr(AL.idx), AL.M.KDE); %Caching, etc. %V6 api.

    AL.M.libsvmmodel = libsvmtrain(CV.Ytr(AL.obs_bits), CV.Xtr(AL.obs_bits,:), P.svmtrainstr);

    elected=true;
end
assert(elected==true);
    
   
function AL = initAL_svm(AL, P, CV)
%Shared SVM init stuff.
AL.hasSVM = true;
AL.M.libsvmmodel = libsvmtrain(CV.Ytr(AL.obs_bits), CV.Xtr(AL.obs_bits,:), P.svmtrainstr); %Now build in!
if(AL.nSeenC>1)
    [tmp, tmp, AL.tr.post] = svmpredict(CV.Ytr, CV.Xtr, AL.M.libsvmmodel, ['-b 1', P.svmteststr]);
else
    AL.tr.post = ones(P.Ntr,1);
end

[AL.tr.slabels, accuracy, tmp] = svmpredict(CV.Ytr, CV.Xtr, AL.M.libsvmmodel, P.svmteststr);

AL.tr.srlog(1)=accuracy(1);
[AL.te.slabels, accuracy, AL.te.post] = svmpredict(CV.Yte, CV.Xte, AL.M.libsvmmodel, P.svmteststr);
AL.te.srlog(1)=accuracy(1);

AL.tr.labels = AL.tr.slabels;
AL.te.labels = AL.te.slabels;
AL.M.optimize = false;
AL.M.C = 1;
AL.M.g = 1/P.Nx;


function AL = initAL_kde4(AL, P, CV, S_c)
%Shared KDE init stuff.

AL.hasKDE = true;

AL.M.KDE = util_createKDE6(CV.Xtr(AL.obs_init,:),CV.Ytr(AL.obs_init),'full',S_c(1),32,numel(unique(CV.Ytr))+1);

AL.lastClass = CV.Ytr(AL.obs_bits);
[AL.tr.knp2,AL.tr.klabels,AL.tr.kpost,AL.M.KDE] = util_inferKDE6(CV.Xtr,AL.M.KDE,0,0,1);
[AL.te.knp2,AL.te.klabels,AL.te.kpost,AL.M.KDE] = util_inferKDE6(CV.Xte,AL.M.KDE);

AL.tr.post = AL.tr.kpost;
AL.te.post = AL.te.kpost;
AL.tr.labels = AL.tr.klabels;
AL.te.labels = AL.te.klabels;


function AL = initAL__SVM_Random(AL,P,CV)
AL.tag = '__SVM_Rand';
%AL.querytag = 'Rand'
AL = initAL_svm(AL, P, CV);

function AL = initAL__SVM_Marg(AL,P,CV)
AL.tag = '_SVM_Marg';
AL = initAL_svm(AL, P, CV);

function AL = initAL_KDE_pWrong(AL, P, CV,S_c,pX,adaptFlag,greedyFlag)
% function AL = initAL_KDE_pWrong(AL, P, CV,S_c,pX,adaptFlag,greedyFlag)
AL.tag = 'KDE__pWrong';

AL = initAL_kde4(AL, P, CV,S_c);  
AL.M.pX_Ycache = zeros(size(pX,1), P.Ny+1);
AL.M.pX_Ycache(:,1) = 1;
  

if(nargin>5 && adaptFlag)
    AL.M.adaptAlpha = true;
    AL.variant = [AL.variant, '_AA'];
else
    AL.M.adaptAlpha = false;
    AL.variant = [AL.variant, '_FA'];
end

AL.M.alpha_init = 0.05;
AL.M.dpalpha = 2;
if(nargin>6 && greedyFlag)
    AL.M.greedy = true;
    AL.variant = [AL.variant, '_CG'];
else
    AL.M.greedy = false;
    AL.variant = [AL.variant, '_CS'];
end
AL.M.KDE_prior = 0;
AL.M.pX = pX;

function AL = initAL_KDE_NPB(AL, P, CV, S_c, pX, priorFlag, adaptFlag, balFlag, greedyFlag, hardFlag, maxSearch, searchCrit)
%function AL = initAL_KDE_NPB(AL, P, CV, S_c, pX, priorFlag, adaptFlag, balFlag, greedyFlag, hardFlag, maxSearch, searchCrit)
% Setup an EE DP AL model.
% Inputs:
%   balFlag:    Blanced or absolute accuracy.
%   maxSearch:  # points to consider with full model
%	searchCrit: 'rand' or 'pWrong' criteria for selecting points ^.
AL.tag = 'KDE__NPB';

%AL = initAL_kde3(AL, P, CV,S_c);    
AL = initAL_kde4(AL, P, CV,S_c);     %V6 kde.
AL.M.pX_Ycache = zeros(size(pX,1), P.Ny+1);
AL.M.pX_Ycache(:,1) = 1;

if(nargin>10 && maxSearch) %How many to subsample consideration each iteration.
    AL.M.maxSearch = maxSearch; 
    if(nargin>11), assert(strcmp(searchCrit,'rand')||strcmp(searchCrit,'pWrong')); end
    if(nargin>11 && strcmp(searchCrit,'pWrong'))        
        AL.M.searchCrit = searchCrit;
        AL.variant = [AL.variant, '_SPwr'];
    else
        AL.M.searchCrit = 'rand';
        AL.variant = [AL.variant, '_SRnd'];
    end
else
    AL.M.maxSearch = 0;
    AL.variant = [AL.variant, '_SEx'];
end

if(nargin>9 && hardFlag)
    AL.M.hardFlag = true;
    AL.variant = [AL.variant, '_CHrd'];
else
    AL.M.hardFlag = false;
    AL.variant = [AL.variant, '_CSft'];
end

if(nargin>7 && balFlag)
    AL.M.balFlag = true;
    AL.variant = [AL.variant, '_CBal'];
else
    AL.M.balFlag = 0;
    AL.variant = [AL.variant, '_CAac'];
end
if(nargin>8 && greedyFlag)
    AL.M.greedy = true;
    AL.variant = [AL.variant, '_CGrd'];
else
    AL.M.greedy = false;
    AL.variant = [AL.variant, '_CSmp'];
end

if(nargin>6 && adaptFlag)
    AL.M.adaptAlpha = true;
    AL.variant = [AL.variant, '_AAd'];
else
    AL.M.adaptAlpha = false;
    AL.variant = [AL.variant, '_AFx'];
end
if(nargin>5 && priorFlag)
    AL.M.KDE_prior = true;
    AL.variant = [AL.variant, '_KPr'];
else
    AL.M.KDE_prior = 0;
    AL.variant = [AL.variant, '_KFx'];
end

AL.M.alpha_init = 0.05;
AL.M.dpalpha = 2;

AL.M.pX = pX;

function AL = initAL_KDE__Lik(AL,P,CV,S_c)
AL.tag = 'KDE__Lik';
%AL = initAL_kde(AL, P, CV,S_c);
AL = initAL_kde4(AL, P, CV,S_c);
AL.M.alpha_init = 0.05;
AL.M.greedy = true;
AL.M.KDE_prior = 1;


function AL = initAL_KDE_SVM_Fusion5(AL,P,CV,S_c)
AL.tag = 'KDE_SVM_Fusion5';
AL.hasFusion = true;
AL = initAL_kde4(AL, P, CV,S_c);
AL = initAL_svm(AL, P, CV);
%AL.M.g_max = P.fusion_g1max;
AL.M.fusion_beta  = P.fusion_beta;
AL.M.k_alg = 2;
AL.M.w     = ones(1,AL.M.k_alg)/AL.M.k_alg;
AL.M.round = 1;
AL.M.g_max = 0;
AL.M.EEC   = 2;
AL.M.alpha_init = 0.05;
AL.M.switchPost = 1;
AL.M.USE_ENT = 1;
AL.M.KDE_prior = 1;
ll = unique(AL.tr.labels);  %Observed classes.
%lh = hist(AL.tr.labels,ll); %Hist of observed classes only.
%cperc = lh(ll)/P.Ntr;       %Inferred classes as %
lh = hist(AL.tr.labels,AL.seenC); %Hist of observed classes only.
cperc = lh/P.Ntr;       %Inferred classes as %
if(AL.nSeenC==1)
    AL.M.H = 0;
else
    AL.M.H = -sum(cperc.*logb(cperc,AL.nSeenC)); %Entropy
end
AL.M.greedy = 0;
AL.M.Ghat = zeros(AL.M.k_alg,1);   %Track the reward gained.
AL.R.ghlog = zeros(0, 2);
AL.R.fwlog = zeros(0, 2);
        
function [idx,labelsE,obs,pidx,KDE] = ElectPointEEDP6(AL, P, x, labelsE, obs, truelabels)
% Elect a point using the EEDP family of models. For V5 KDE api.
    N = numel(labelsE);
    pX = AL.M.pX;
    alpha = AL.M.dpalpha;
    Nc = P.Ny;
    %pX_Ycache = AL.tr.knp2;
    opts.alpha = AL.M.dpalpha;
    
    %% Do an inference.
    pC_X   = zeros(N, Nc+2);
    %[pX_C6,~,~,KDE6] = util_inferKDE6(x,KDE6,0,1,1,1,1); %Get dataset likelihood under each class.
    [pX_C,~,~,KDE] = util_inferKDE6(x,AL.M.KDE,0,1,1,1,1); %Get dataset likelihood under each class.

    pC_X(:,1:Nc+1)  = pX_C .* repmat((0+[AL.M.KDE.K_c{:,2}]) / (sum(0+[AL.M.KDE.K_c{:,2}]) + alpha), N,1);
    pC_X(:,end)     = pX * alpha / (sum(0+[AL.M.KDE.K_c{:,2}]) + alpha);
    pC_X = normalise(pC_X,2); %Work out p(Classes, Unseen | Dataset).
    
    %% (Intelligent) Subsampling.
    if(AL.M.maxSearch) %Random list of unseen.
        nUnseen = sum(~obs);
        if(strcmp(AL.M.searchCrit,'rand'))         
            ridx = randperm(nUnseen);
            ridx = ridx(1:min(AL.M.maxSearch,nUnseen));
            lUnseen = find(~obs);
            lUnseenSel = lUnseen(ridx);
            ridxBits = false(1,N);            
            ridxBits(lUnseenSel) = true;
            ridx = find(ridxBits);
        elseif(strcmp(AL.M.searchCrit,'pWrong'))
            [~,pidx] = util_evalPwrong2(x,KDE,obs,opts,pX);
            %[~,pidx] = util_evalPwrong(x,AL.M.KDE,obs,opts,pX,pX_Ycache,AL.lastClass);
            [~,sidx]=sort(pidx,'descend');
            ridx = sidx(1:min(AL.M.maxSearch,nUnseen));
        end
    else
        ridx = find(~obs');%Or all unseen in order.
        %ridxBits = ~obs;
    end
    
    %% Actually do the expectation.
    pidx = zeros(N,1);    
    for i = ridx
        obs(i) = true;            
            
        pidx(i) = util_evalExpAcc3c(i, x, truelabels, KDE, pX, obs, opts, pC_X(i,:)); %V6 API.
            
        obs(i) = false;
    end
    
    %% Select / Label.
    pidx = normalise(pidx);
    if(AL.M.greedy)
        [~,idx] = max(pidx);
    else
        idx = sum(cumsum(normalise(pidx))<rand) + 1;
    end
    assert(~obs(idx));
    obs(idx) = true;
    labelsE(idx) = truelabels(idx);   

function [idx,labelsE,obs,pidx] = ElectPointLik(labelsE, obs, truelabels, np)
%% Select point with minimum p(X|C) for all C.
[tmp,c_map] = max(np,[],2);
ind=sub2ind(size(np),(1:size(np,1))',c_map);
pX = np(ind);

pidx = 1-(pX-min(pX))/(max(pX)-min(pX));
pidx(obs) = 0;

pX(obs') = inf;
[tmp,idx] = min(pX);

obs(idx)  = true;
labelsE(idx) = truelabels(idx);

function [idx,labelsE,obs,pidx] = ElectPointEntropy(labelsE, obs, truelabels, post, b)
%Entropy method.
if(b==1)
    pidx = ones(size(labelsE));
    pidx(obs) = 0;
    pidx = normalise(pidx);
    idx = sum(cumsum(pidx)<rand)+1; %More numerically stable!? :-/.
    obs(idx) = true;
    return;
end
if(nargin<5)
    H = -sum(post.*log(post),2);
else
    H = -sum(post.*logb(post,b),2);    
end
%Hack numerical :-(.
zerobits = isnan(H);
if(any(zerobits))
    fprintf(1,'Warning replacing %d entropy nans with zeros\n',sum(zerobits));
    H(zerobits) = 0;
end

pidx = (H-min(H))/(max(H)-min(H));
pidx(obs) = 0;

H(obs) = -inf;
[tmp,idx]=max(H);
obs(idx)  = true;
labelsE(idx) = truelabels(idx);


function [idx,labelsE,obs,pidx] = ElectPointRandom(labelsE, obs, truelabels)
%Post Prob.
pidx = ~obs/sum(~obs);
idx = sum(cumsum(pidx)<rand)+1; %More numerically stable!? :-/.
obs(idx)  = true;


pidx = 0.05*(pidx>0); %For the benefit of fusers!

%Answer.
labelsE(idx) = truelabels(idx);