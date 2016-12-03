%% Summarize the results of 'batch_experiment.m'
dl = {'glass','yeast','covertype'};
lab = {'Fusion/TKDE12','Likelihood','DPEA/ECCV12','pWrong/BMVC11','SVM UNC','Random'};
ND = numel(dl);
clist = {'k','b','g','r','c','b--','m','m--','b:','k:'}; nM = numel(clist);
dataset = 'All';
DR = zeros(ND, nM); %Discovery area.
CR = zeros(ND, nM); %Classify area.
AT = zeros(ND, nM); %Asymptotic classify.
AD = zeros(ND, nM); %Asymptotic discovery.
DISP = 1;
%%
for d = 1:ND
    if(DISP), figure(d); clf; clf;  set(gcf,'Name',[dl{d},' Class/Discov T8']);end
    tc2 = dir(['results/',dl{d},'/g_npb3/*.mat']);
    tc2 = {tc2.name};
    str = ['results/',dl{d},'/g_npb3/',tc2{1}];
    tmp = load(str);  nTrials=size(tmp.R.UCLOG,1);
    
    j=0;    
    fprintf(1,'--Dataset: %s. %d Trials.--\n', dl{d}, nTrials);%, nTrials: %d, nAiter: %d\n',dl{d}, nTrials, nAiter);

    for m = 1:6
        j=j+1;
        str = ['results/',dl{d},'/g_npb3/',tc2{m}];
        RES(m) = load(str); P = RES(m).P;
        %Trials per model---
        [nTrials,nAiter] = size(RES(m).R.UCLOG);
        la = 1:min(nAiter,200); 
        lt = 1 : nTrials;
        %Trials per model---
        DR(d,m) = sum(sum(RES(m).R.UCLOG(lt,la)))/(nAiter*nTrials*RES(m).P.Ny);
        CR(d,m) = sum(sum(RES(m).R.te.ASRLOG(lt,la)))/(nAiter*nTrials);
        AT(d,m) = sum(RES(m).R.te.ASRLOG(lt,la(end)))/nTrials;
        AD(d,m) = sum(RES(m).R.UCLOG(lt,la(end)))/(nTrials*RES(m).P.Ny);
        if(DISP)
            figure(d); 
            subplot(2,2,1); hold on; plot(1:nAiter, mean(RES(m).R.UCLOG(lt,:),1)/P.Ny ,clist{j}); title('Discover');
            subplot(2,2,2); hold on; plot(1:nAiter, mean(RES(m).R.tr.ASRLOG(lt,:),1),clist{j}); title('Train');
            subplot(2,2,4); hold on; plot(1:nAiter, mean(RES(m).R.te.ASRLOG(lt,:),1),clist{j}); title('Test');
            fprintf(1,'%d AUTR: %1.3f, AUTE: %1.3f (%s, %s) \n', m, DR(d,m), CR(d,m), clist{j}, lab{m});
        end
    end
    
    if(DISP),
        figure(d); subplot(2,2,2); legend(lab);
    end    
    
end

%%
avg=mean(CR,1);
[~,w]=max(CR,[],2); h=hist(w,1:6);
disp('-- Summary --');
for m = 1 : 6
    fprintf(1,'%15s. Avg: %0.2f, Win: %02d\n', lab{m}, avg(m), h(m));
end
