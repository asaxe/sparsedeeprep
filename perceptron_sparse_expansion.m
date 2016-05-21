%clear

alpha = 1/100;
Nepochs = 1000;

Ns = 50;
P = 50;
dS = .05;
Ncs = [100 1000 10000];

c = rand(Ns,P)>.5;
y = 2*((rand(1,P)>.5)-1/2);

for n = 1:length(Ncs)
    Nc = Ncs(n);
    
    J = randn(Nc,Ns);
    w = zeros(1,Nc);


    for i = 1:5000

        x = (2*(c-1/2).*(2*((rand(Ns,P) >= (dS/2)) - 1/2)))/2+1/2;

        h = J*x;
        s = h>0;

        yh = 2*((w*s>0)-1/2);

        e = (y-yh)/2;

        err(i,n) = mean(abs(e));
        
        w = w + alpha*(y-yh)*h';
        wn(i,n) = norm(w);

    end
end
%%
hold on
smoothed_err = conv2(err, ones(10000,1)/10000,'valid')
plot(smoothed_err)
for t = 1:length(Ncs)
   leg_label{t} = num2str(Ncs(t)); 
end
legend(leg_label)