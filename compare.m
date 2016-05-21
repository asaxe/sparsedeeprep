%ds = 'mnist'; trnlen = 10000;
%ds = 'mnist_background_random'; trnlen = 10000;
%ds = 'mnist_background_images'; trnlen = 10000;
%ds = 'convex'; trnlen = 6000;
ds = 'rectangles'; trnlen = 1000;
%ds = 'rectangles_im'; trnlen = 10000;

tmp = dlmread(sprintf('%s_train.amat', ds));
trn = single(tmp(1:trnlen,1:end-1)');
trn_lab = tmp(1:trnlen,end)';

tmp = dlmread(sprintf('%s_test.amat', ds));
tst = single(tmp(1:50000,1:end-1)');
tst_lab = tmp(1:50000,end)';

m = mean(trn(:)); trn = trn - m; tst = tst - m;

clear tmp;

trn = bsxfun(@rdivide, trn, sqrt(sum(trn.^2)));
tst = bsxfun(@rdivide, tst, sqrt(sum(tst.^2)));

clip = @(K)max(min(K,1),-1);

ssrelu = @(K)(sqrt(1-K.^2) + (pi-acos(K)).*K)/pi;
tsrelu = @(K)(1 - acos(K)/pi).*K;

func = { ssrelu ,  tsrelu };
name = {'SSReLU', 'TSReLU'};

for c = 10.^[0 1 2 3]
    
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
            
            if res(1) >= acc, acc = res(1); else break; end % auto stop
        end
        
        fprintf('\n');
    end
end

