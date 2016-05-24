clear
%% Generate dataset
Nepochs = 1;
Ns = 50; % Input dimension
P = 50; % Number of centroids
fprintf('Num examples = %d\n\n',P*Nepochs) % = 10; % Number of epochs of training data to generate for overall batch. Total # of training samples in batch is P*Nepochs
Nepochs_test = 10;
dS = .3; % Probability of noise corruption
Ncs = [1000]; % Number of random hidden layer filters

c = rand(Ns,P)>.5; % Centroid locations
yl = 2*((rand(1,P)>.5)-1/2); % Centroid labels

x = (2*(repmat(c,1,Nepochs)-1/2).*(2*((rand(Ns,P*Nepochs) >= (dS/2)) - 1/2)));%/2+1/2;
y = repmat(yl,1,Nepochs);

xt = (2*(repmat(c,1,Nepochs_test)-1/2).*(2*((rand(Ns,P*Nepochs_test) >= (dS/2)) - 1/2)));%/2+1/2;
yt = repmat(yl,1,Nepochs_test);

%% Tanh network
clear W loss err c_ts err_ts

% Params
g = .2;
Nhid = [500 500 500 500];
Nitr = 8000;
alpha = .0005/2;
err_thresh = .5;

err_fn = @(yhat,yact) 1-mean([(yhat(yact(:)==1) > err_thresh); (yhat(yact(:)==0) < 1-err_thresh)]);

sz = [size(x,1) Nhid size(y,1)];

% Init weights
Nl = length(sz);

% Initialize weights (scaling here is important)

for i = 1:Nl-1
    W{i} = g*normrnd(0, 1/sqrt((sz(i)+sz(i+1))/2), sz(i+1), sz(i));
end

for itr = 1:Nitr
    
    % Draw minibatch
    x = (2*(repmat(c,1,Nepochs)-1/2).*(2*((rand(Ns,P*Nepochs) >= (dS/2)) - 1/2)));%/2+1/2;
    y = repmat(yl,1,Nepochs);


    [loss(itr),yhat,grad] = multilayer_tanh(W,x,y);
    err(itr) = err_fn(yhat,y);
    %if mod(itr,6000)==0
    %   alpha = alpha/1.2; 
    %end
    
    if mod(itr,100)==0
        itr
        
        N=100;
        
%         plot((1:length(loss))*size(x,2),loss/size(y,2),'linewidth',2)
%         xlabel('Nexamples')
%         ylabel('Loss')
        %hline(c_ts)
        
        tmp = conv(err,ones(1,N)/N,'valid');
        plot((1:length(tmp))*size(x,2),tmp,'linewidth',2)
        xlabel('Nexamples')
        ylabel('Error (%)')
        %hline(err_ts)
        drawnow
    end
    
    for l = 1:length(W)
        W{l} = W{l} - alpha*grad{l};
    end
end

save(sprintf('dres%d_ds%g.mat',length(Nhid),dS),'err')
%%
[~,yhat,~] = multilayer_tanh(W,x,y);

%%
subplot(121)
plot(1:Nitr,c/size(ytr,2),1:Nitr,c_ts/size(yts,2),'linewidth',2)
%hline(c_ts)
subplot(122)
plot(1:Nitr,err,1:Nitr,err_ts,'linewidth',2)
%hline(err_ts)

%%
dS=.3;
maxd =4;
colors = [linspace(0,.6,maxd)' linspace(0, .6,maxd)' linspace(.8,1,maxd)'];
colors = flipud(colors);
clear leg_txt

for d = 1:maxd
    N = 100;
    load(sprintf('dres%d_ds%g.mat',d,dS))
    tmp = conv(err,ones(1,N)/N,'valid');
    plot((1:length(tmp))*50,tmp,'Color',colors(d,:),'linewidth',2)
    hold on
    xlabel('Nexamples')
    ylabel('Error (%)')
    
    leg_txt{d} = sprintf('D=%d',d);
end

legend(leg_txt)