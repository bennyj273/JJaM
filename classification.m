k = 15;

%KNNModel = fitcknn(pcaR, isMoving, 'NumNeighbors', k) %2 seconds now
Mdl = TreeBagger(50, pcaR, isMoving)

est_motion = predict(Mdl, pcaR) 
for i = 1:size(est_motion,1)
    est
end
%This is the bottleneck here - O(nf), where n=datapoints, f=features
%It takes about 60 seconds per run, so this should be 15 minutes
toc
plot(est_motion)
hold on
smoothed = conv(est_motion, fil, 'same')
plot(smoothed)
plot(output{3,5})
title(num2str(k) + " " + num2str(corr(output{3,5}(1:5996), smoothed)))
