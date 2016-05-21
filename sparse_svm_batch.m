addpath('libsvm-compact-0.1')
addpath('libsvm-compact-0.1/matlab')


%% Generate dataset

Ns = 50; % Input dimension
P = 50; % Number of centroids
Nepochs = 1000; % Number of epochs of training data to generate for overall batch. Total # of training samples in batch is P*Nepochs
dS = .05; % Probability of noise corruption
Ncs = [1000]; % Number of random hidden layer filters

c = rand(Ns,P)>.5; % Centroid locations
y = 2*((rand(1,P)>.5)-1/2); % Centroid labels

x = (2*(repmat(c,1,Nepochs)-1/2).*(2*((rand(Ns,P*Nepochs) >= (dS/2)) - 1/2)))/2+1/2;
y = repmat(y,1,Nepochs);

%% Train SVM

trn = single(x);
trn_lab = y;

tst = single(x);
tst_lab = y;

m = mean(trn(:)); trn = trn - m; tst = tst - m;

clear tmp;

trn = bsxfun(@rdivide, trn, sqrt(sum(trn.^2)));
tst = bsxfun(@rdivide, tst, sqrt(sum(tst.^2)));

clip = @(K)max(min(K,1),-1);

ssrelu = @(K)(sqrt(1-K.^2) + (pi-acos(K)).*K)/pi;
tsrelu = @(K)(1 - acos(K)/pi).*K;

func = { ssrelu ,  tsrelu };
name = {'SSReLU', 'TSReLU'};

for c = 10.^[1]
    
    for f = [2 1] %{tsrelu, ssrelu}

        ktrn = zeros(size(trn,2)+1, size(trn,2), 'single'); ktrn(1,:) = 1:size(trn,2);
        ktst = zeros(size(trn,2)+1, size(tst,2), 'single'); ktst(1,:) = 1:size(tst,2);

        ktrn(2:end,:) = clip(trn' * trn);
        ktst(2:end,:) = clip(trn' * tst);

        acc = 0;
        
        for l = 1:100

            fprintf('%s: c=%.1e, l=%02d, ', name{f}, c, l);

            ktrn(2:end,:) = clip(func{f}(ktrn(2:end,:)));
            ktst(2:end,:) = clip(func{f}(ktst(2:end,:)));

            [~, res, ~] = svmpredict_inplace(tst_lab, ktst, svmtrain_inplace(trn_lab, ktrn, sprintf('-t 4 -c %d -q', c)));
            
            if res(1) > acc , acc = res(1); else break; end % auto stop
        end
        
        fprintf('\n');
    end
end

