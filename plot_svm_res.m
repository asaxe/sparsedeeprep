
%% dS = .5
clear
title_l = 'dS=.5';
Nex = [500 5000 10000 25000];

acc{1} = [75.604 76.496 76.568 76.524];
acc{2} = [85.116 86.384 87.04 87.484 87.664 87.744 87.808 87.888 87.888];
acc{3} = [87.596 89.008 89.688 89.884 90.048 90.088 90.184 90.176];
acc{4} = [88.168 90.412 90.764 90.956 90.972 90.972];

%% dS = .1
clear
title_l = 'dS=.1';
Nex = [500 5000 10000 25000];
    
acc{1} = [98.316 98.616 98.78 98.88 98.896 98.916 98.924 98.896];
acc{2} = [99.572 99.764 99.832 99.852 99.86 99.872 99.876 99.88 99.876];
acc{3} = [99.74 99.86 99.892 99.908 99.912 99.916 99.916];
acc{4} = [99.808 99.892 99.928 99.936 99.94 99.94];

%% dS = .05
clear
title_l = 'dS=.05';
Nex = [500 5000 10000 15000];

acc{1} = [94.08 98.356 99.292 99.58 99.804 99.892 99.932 99.956 99.964 99.972 99.976 99.976];
acc{2} = [96.404 99.348 99.844 99.928 99.964 99.976 99.988 99.992 100 100];
acc{3} = [97.116 99.148 99.66 99.808 99.872 99.916 99.944 99.96 99.96];
acc{4} = [100 100 100 100 100 100 100 100 100 100 100 100];




%% By depth
colors = [linspace(0,.6,length(Nex))' linspace(0, .6,length(Nex))' linspace(.8,1,length(Nex))'];
colors = flipud(colors);
clear leg_txt

%subplot(122)
for e = 1:length(Nex)
   
   plot(1:length(acc{e}),1-acc{e}/100,'-o','Color',colors(e,:),'linewidth',3)
   hold on
   leg_txt{e} = sprintf('Nex=%d',Nex(e));
end
%ylim([0 .3])
legend(leg_txt,'location','NorthEast')
xlabel('Depth')
ylabel('Test Error (%)')
title(title_l)


%% By Nex

maxdepth = 0;
for e = 1:length(Nex)
    maxdepth = max(maxdepth,length(acc{e}))
end
for d = 1:maxdepth
   acc_nexv{d}=[];
   acc_nex{d}=[];
end
for e = 1:length(Nex)
    for d = 1:maxdepth
        length(acc{e})
        
        if length(acc{e}) >= d
            acc_nexv{d} = [acc_nexv{d} Nex(e)];
            acc_nex{d} = [acc_nex{d} acc{e}(d)];
        end
    end
end

colors = [linspace(.8,1,maxdepth)' linspace(0,.6,maxdepth)' linspace(0, .6,maxdepth)'];
colors = flipud(colors);


for d = 1:maxdepth
   plot(acc_nexv{d},1-acc_nex{d}/100,'Color',colors(d,:),'linewidth',3)
   hold on
   leg_txt{d} = sprintf('D=%d',d');
end
legend(leg_txt,'location','NorthEast')
xlabel('# of examples')
ylabel('Test Error (%)')
title(title_l);

