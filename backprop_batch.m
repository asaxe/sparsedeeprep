clear
%% Generate dataset
Nepochs = 10;
Ns = 50; % Input dimension
P = 50; % Number of centroids
fprintf('Num examples = %d\n\n',P*Nepochs) % = 10; % Number of epochs of training data to generate for overall batch. Total # of training samples in batch is P*Nepochs
Nepochs_test = 10;
dS = .5; % Probability of noise corruption
Ncs = [1000]; % Number of random hidden layer filters

c = rand(Ns,P)>.5; % Centroid locations
yl = 2*((rand(1,P)>.5)-1/2); % Centroid labels

x = (2*(repmat(c,1,Nepochs)-1/2).*(2*((rand(Ns,P*Nepochs) >= (dS/2)) - 1/2)));%/2+1/2;
y = repmat(yl,1,Nepochs);

xt = (2*(repmat(c,1,Nepochs_test)-1/2).*(2*((rand(Ns,P*Nepochs_test) >= (dS/2)) - 1/2)));%/2+1/2;
yt = repmat(yl,1,Nepochs_test);

%% Tanh network
clear W c err c_ts err_ts

% Params
g = .2;
Nhid = [500 500 500];
Nitr = 2500;
alpha = .00005;
err_thresh = .9;

err_fn = @(yhat,yact) 1-mean([(yhat(yact(:)==1) > err_thresh); (yhat(yact(:)==0) < 1-err_thresh)]);

sz = [size(x,1) Nhid size(y,1)];

% Init weights
Nl = length(sz);

% Initialize weights (scaling here is important)

for i = 1:Nl-1
    W{i} = g*normrnd(0, 1/sqrt((sz(i)+sz(i+1))/2), sz(i+1), sz(i));
end

for itr = 1:Nitr

    [c(itr),yhat,grad] = multilayer_tanh(W,x,y);
    err(itr) = err_fn(yhat,y);
    
    [c_ts(itr),yhat,~]= multilayer_tanh(W,xt,yt);
    err_ts(itr) = err_fn(yhat,yt);
    
    if mod(itr,10)==0
        itr
        subplot(121)
        plot(1:length(c),c/size(y,2),1:length(c_ts),c_ts/size(yt,2),'linewidth',2)
        %hline(c_ts)
        subplot(122)
        plot(1:length(err),err,1:length(err_ts),err_ts,'linewidth',2)
        %hline(err_ts)
        drawnow
    end
    
    for l = 1:length(W)
        W{l} = W{l} - alpha*grad{l};
    end
end

%%
[~,yhat,~] = multilayer_tanh(W,x,y);

%%
subplot(121)
plot(1:Nitr,c/size(ytr,2),1:Nitr,c_ts/size(yts,2),'linewidth',2)
%hline(c_ts)
subplot(122)
plot(1:Nitr,err,1:Nitr,err_ts,'linewidth',2)
%hline(err_ts)
