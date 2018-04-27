function [v] = MovingWinFeats(x, fs, winLen, winDisp, featFn)
    numWins = floor(((length(x)/fs) - winLen)/winDisp);
    v = zeros(numWins, 1);
    winDispSamples = winDisp*fs;
    winLenSamples = winLen*fs;
    for i = 1:numWins
        sample = x(i*winDispSamples:i*winDispSamples+winLenSamples);
        v(i) = featFn(sample);
    end
end