clear

alpha = 1/10000;

%% Generate dataset
Ns = 50; % Input dimension
P = 50; % Number of centroids
Nepochs = 10; % Number of epochs of training data to generate for overall batch. Total # of training samples in batch is P*Nepochs
dS = .05; % Probability of noise corruption
Ncs = [2000]; % Number of random hidden layer filters

c = rand(Ns,P)>.5; % Centroid locations
y = 2*((rand(1,P)>.5)-1/2); % Centroid labels

x = (2*(repmat(c,1,Nepochs)-1/2).*(2*((rand(Ns,P*Nepochs) >= (dS/2)) - 1/2)))/2+1/2;
y = repmat(y,1,Nepochs);

%% Run batch perceptron learning
for n = 1:length(Ncs)
    Nc = Ncs(n);
    
    J = randn(Nc,Ns);
    w = zeros(1,Nc);



    h = J*x;
    s = h>0;
    
    clear err
    for i = 1:10000

        i

        yh = 2*((w*s>0)-1/2);

        e = (y-yh)/2;

        err(i,n) = mean(abs(e));
        
        w = w + alpha*(y-yh)*h';
        wn(i,n) = norm(w);

    end
end
%%
hold on
smoothed_err = conv2(err, ones(10,1)/10,'valid')
plot(1-smoothed_err)
for t = 1:length(Ncs)
   leg_label{t} = num2str(Ncs(t)); 
end
legend(leg_label)