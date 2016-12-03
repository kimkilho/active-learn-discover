%% Batch script to run a bunch of datasets.

%Cell arraoy of <filename>s for .mat files found in data/<filename>.
elist = {'glass','yeast','covertype'};

%How many active learning iterations to run each mat file for.
ilist = [150,150,150];

%Which models to run?
% 1. Random criteria, SVM classifier.
% 2. Entropy criteria, SVM classifier.
% 3. Likelihood criteria, KDE classifier.
% 4. pWrong criteria, KDE classifier. (Haines 2011 BMVC)
% 5. Gen/Disc fusion criteria, SVM/KDE classifier. (Hospedales 2012 IEEE TKDE)
% 6. DPEA criteria, KDE classifier. (Hospedales 2012 ECCV)

domodel = [1:6];

for j = [1,2] %Crossvalidation Fold.
    for i =  1 : numel(elist) %Dataset.
        if(~exist(['results/',elist{i}],'dir'))
            mkdir(['results/',elist{i}]);
        end
        if(~exist(['results/',elist{i},'/g_npb3/'],'dir'))
            mkdir(['results/',elist{i},'/g_npb3/']);
        end    
        fprintf('Run (%s) %d/%d\n', elist{i}, i,numel(elist));
        active_learn_wrapper(elist{i},0,ilist(i),domodel,j);
    end
end
now2=datestr(now);