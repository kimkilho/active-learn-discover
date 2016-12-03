Active Learning and Discovery Package V0.1
Timothy Hospedales, 09/2012.
http://www.eecs.qmul.ac.uk/~tmh/

Described in: [Hospedales et al. PAKDD'11 / Haines et al. BMVC'11, Hospedales et al. IEEE TKDE'12 and Hospedales et al. ECCV'12]. 

See batch_experiment.m for an example run of a few UCI datasets. (Note: This should take about 10 minutes to run depending on speed of your PC)

Models included:
% 1. Random criteria, SVM classifier.
% 2. Entropy criteria, SVM classifier.
% 3. Likelihood criteria, KDE classifier.
% 4. pWrong criteria, KDE classifier. (Haines 2011 BMVC)
% 5. Gen/Disc fusion criteria, SVM/KDE classifier. (Hospedales 2012 IEEE TKDE)
% 6. DPEA criteria, KDE classifier. (Hospedales 2012 ECCV)

Notes:
- To run models [1,2,5] you may need to compile libsvm for your platform in ./libsvm/. Everything else is matlab. 
- Renamed libsvm's svmtrain -> libsvmtrain to avoid conflict with matlab builtin function svmtrain.

Data Structure Format: To setup the input for the script, please save the following structures in the input mat file.
P.Nx  = Number of input dimensions.
P.Ny  = Number of classes.
P.Ntr = Number of train instances.
P.Nte = Number of test instances.
CV(fold).Xr  = Train data. [P.Ntr x P.Nx]
CV(fold).Xt  = Test data.  [P.Ntr x 1].
CV(fold).Ytr = Train labels.
CV(fold).Yte = Test labels
