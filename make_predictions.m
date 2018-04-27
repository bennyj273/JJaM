function [predicted_dg] = make_predictions(test_ecog)

%
% Inputs: test_ecog - 3 x 1 cell array containing ECoG for each subject, where test_ecog{i} 
% to the ECoG for subject i. Each cell element contains a N x M testing ECoG,
% where N is the number of samples and M is the number of EEG channels.(
% Outputs: predicted_dg - 3 x 1 cell array, where predicted_dg{i} contains the 
% data_glove prediction for subject i, which is an N x 5 matrix (for
% fingers 1:5)
% Run time: The script has to run less than 1 hour.

load jjam_model.mat %loads f_predictors 
% load ecog_test.mat
% test_ecog = ecogTestVect;
ecogLength = length(test_ecog{1})-1;
numChannels = zeros(3,1);
for i = 1:3
    sizeTemp = size(test_ecog{i});
    numChannels(i) = sizeTemp(2);
end

samplingFrequency = 1000;
windowLength = 0.1; %100 ms
overlap = 0.05; %50 ms overlap
windowDisp = windowLength - overlap;

features_ecog = cell(3, max(numChannels), 6); %max cell length

%Functions

%Take average for each sample

avg = @(x) mean(x); %Average of everything in the channel

for i = 1:3 %per subject
    for ch = 1:numChannels(i) %per channel
      features_ecog{i, ch, 1} = [MovingWinFeats(test_ecog{i}(:, ch), 1000, windowLength, windowDisp, avg); 0]; %This seems fake
    end
end

%Evaluate the spectrum at 1000/2 + 1 = 501 frequencies
Fs = 1000;
freqNum = floor(Fs/2) + 1; %We need to have 5-175 Hz - if freqNum is 501, this covers the spectrum

%Frequency bands vary from 0 to 1 pi rad/sample
%Vary from 0 to pi rad/sample * 1000 samples/sec
%Vary from 0 to 1000pi Hz
%0 to 3.1416 Hz
freqbands = [5 10; 15 20; 20 25; 40 60;75 115; 125 160; 160 175; 200 220; 220 240; 240 260];
angfreqbands = freqbands*2*pi();
angfreqpercents = angfreqbands/(Fs*pi()); %As a fraction of 1000pi, the max frequency
angfreqindices = floor(angfreqpercents*freqNum);

%Find frequency bands for each sample
for i = 1:3 %per channel
    for ch = 1:numChannels(i)
        [spec, f, t] = spectrogram(test_ecog{i}(:, ch), windowLength*samplingFrequency, overlap*samplingFrequency, Fs);
        for band = 1:size(freqbands, 1)
            features_ecog{i, ch, band+1} = abs(mean(spec(angfreqindices(band,:), :)))';

        end
    end
end

featLength = length(features_ecog{1,1,2});
predicted_pos_test = cell(3, 1);
for i = 1:3
    feats = [];
    for ch = 1:numChannels(i)
        for f = 1:6
            norm_features_test{i,ch,f} = (features_ecog{i,ch,f}(1:featLength)-means{i,ch,f})/stdevs{i,ch,f};
            sz = 2;
            filt = ones(sz, 1)/sz;
            norm_features_test{i, ch, f} = conv(norm_features_test{i, ch, f}, filt, 'same');
            feats = [feats norm_features_test{i, ch, f}];
        end
    end
    %Features is a feature matrix of 6*channels features
    prediction = [];
    for finger = 1:5
        N = 3;
        M = size(feats,1) - N+1; %Total time bins
        nu = size(feats,2);
        R = zeros(M, 1);
        for j = 1:M
            R(j, 1) = 1;
        end
        for l = 1:nu
            matrix = zeros(M, N);
            for j = 1:M
                for k = 1:N
                    matrix(j, k) = feats(j+k-1, l);
                end
            end
            R = [R matrix]; 
        end
        %Need to spline it back up to 300,000
        est_pos = R*f_predictors{i, finger}; %Make predictions
        x = est_pos(1)*ones(N, 1);
        est_pos = [x; est_pos];
        %Need to spline it back up to 300,000
        est_pos_full = spline(0:50:ecogLength-1, est_pos, 0:1:ecogLength-1);
        prediction = [prediction est_pos_full'];
    end
    predicted_pos_test{i} = prediction;
end

sz = 300;
filt = ones(sz, 1)/sz;
predicted_pos_test_filtered = cell(3,1);
for i = 1:3
    for ch = 1:5
        predicted_pos_test_filtered{i}(:,ch) = conv(predicted_pos_test{i}(1:end, ch), filt, 'same');
    end
end

predpos_test_filtfilt= cell(3,1);
fs = 1000;
band = [0.6 1.3]/(fs/2); %80 - 50 Hz of cutoff
[f, e] = butter(1, band,'stop');
for i = 1:3
    for fing = 1:5
        predpos_test_filtfilt{i}(:,fing) = filtfilt(f, e, predicted_pos_test_filtered{i}(:,fing));
    end
end

predicted_dg = predpos_test_filtfilt;

end

