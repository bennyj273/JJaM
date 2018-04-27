%% Creating cells of Training set dg and ecog data
session_dg = cell(3,1);
session_ecog = cell(3,1);
dg = cell(3, 1);
ecog = cell(3, 1);
ecog_filtered = cell(3, 1);
for i = 1:3
    str_dg = strcat('I521_Sub', num2str(i), '_Training_dg');
    str_ecog = strcat('I521_Sub', num2str(i), '_Training_ecog');
    session_dg{i} = IEEGSession(str_dg, 'jaimiec', '//home/jaimiec/Documents/Spring_2018/BE521/Tutorial/jai_ieeglogin.bin');
    session_ecog{i} = IEEGSession(str_ecog,'jaimiec', '//home/jaimiec/Documents/Spring_2018/BE521/Tutorial/jai_ieeglogin.bin');
end

session_ecog_leaderboard = cell(3,1);
ecog_leaderboard = cell(3, 1);
numChannels = zeros(3,1); %Varies per subject
%Get the sessions
for i = 1:3
    str_ecog = strcat('I521_Sub', num2str(i), '_Leaderboard_ecog');
    session_ecog_leaderboard{i} = IEEGSession(str_ecog, 'jaimiec', '//home/jaimiec/Documents/Spring_2018/BE521/Tutorial/jai_ieeglogin.bin');
    numChannels(i)=length(session_ecog{i}.data.rawChannels);
end

%Get the actual data for each session
for i = 1:3
    ecog_leaderboard{i} = session_ecog_leaderboard{i}.data.getvalues(1:147500, 1:numChannels(i));
end

%Sampling Rate
sampr = 1000;
%Get duration in seconds (Leaderboard)
durationInUSec_train = session_ecog{1}.data.rawChannels(1).get_tsdetails.getDuration;
durationInSec_train = durationInUSec_train/(10^6); %uS -> S conversion
%Get duration in seconds (Training)
durationInUSec_lbd = session_ecog_leaderboard{1}.data.rawChannels(1).get_tsdetails.getDuration;
durationInSec_lbd = durationInUSec_lbd/(10^6); %uS -> S conversion


%Number of values (Leaderboard)
num_values_train = durationInSec_train*sampr +1; 
%Number of values (Training)
num_values_lbd = durationInSec_lbd*sampr +1; 

for i = 1:3
    dg{i} = session_dg{i}.data.getvalues(1:num_values_train, 1:5);
    ecog{i} = session_ecog{i}.data.getvalues(1:num_values_train, 1:numChannels(i));
end

%% Get values
for i = 1:3
    dg{i} = session_dg{i}.data.getvalues(1:270000, 1:5);
    ecog{i} = session_ecog{i}.data.getvalues(1:270000, 1:numChannels(i));
end


%% Calculate mean features - training
samplingFrequency = 1000;
windowLength = 0.1; %100 ms
overlap = 0.5*windowLength; %50 ms overlap
windowDisp = windowLength - overlap;

fingers = 5; 
features = cell(3, max(numChannels), 11); %max cell length - adding a correlation band
output = cell(3, 5);
%Features = 1 mean, 2-6 the 5 frequency bands

%Functions

thresh = 15;
%Take average for each sample
mn = @(x) mean(norm(x)) > thresh;
avg = @(x) mean(x)

%Moving vs not moving

windowLength = 0.1;
for i = 1:3 %per subject
    for f = 1:fingers
        output{i, f} = [];
        for j = 1:windowLength*samplingFrequency:size(dg{i}(:, f), 1)-windowLength*samplingFrequency
            sample = dg{i}(j:j+windowLength*samplingFrequency, f);
            output{i, f} = [output{i, f} mean(norm(sample)) > thresh];
        end
    end
    for ch = 1:numChannels(i) %per channel
        features{i, ch, 1} = [MovingWinFeats(ecog{i}(:, ch), 1000, windowLength, windowDisp, avg); 0]; %This seems fake
    end
end

%% Grouping into moving vs not moving


%% Calculate frequency features - training
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

%Size spectrogram = 501 * 2949

%Find frequency bands for each sample
for i = 1:3 %per channel
    for ch = 1:numChannels(i)
        [spec, ~, t] = spectrogram(ecog{i}(:, ch), windowLength*samplingFrequency, overlap*samplingFrequency, Fs);
        for band = 1:size(freqbands,1)
            features{i, ch, band+1} = abs(mean(spec(angfreqindices(band,:), :)))';
        end
    end
end

%% Put correlation features at the end

%% Decimation - training
dg_subsampled = cell(3, 1);
for i = 1:3
    decimated = [];
    for finger = 1:5
        decimated = [decimated decimate(dg{i}(:, finger), 50)];
    end
    dg_subsampled{i} = decimated;
end

%% Prediction - training
predicted_pos = cell(3, 5);
f_predictors = cell(3, 5);
means = cell(3, 5,9);
stdevs = cell(3,5,9);
BETAS = cell(3, 5);
Mdls = cell(3, 5);

for i = 1:3
    feats = [];
    for ch = 1:numChannels(i)
        for f = 1:size(features,3) %number of features
            means{i,ch,f} = mean(features{i,ch,f});
            stdevs{i,ch,f} = std(features{i,ch,f});
            norm_features{i,ch,f} = (features{i,ch,f}-means{i,ch,f}) / stdevs{i,ch,f};
            sz = 2;
            filt = ones(sz, 1)/sz;
            norm_features{i, ch, f} = conv(norm_features{i, ch, f}, filt, 'same');
            size(norm_features{i, ch, f})
            feats = [feats norm_features{i, ch, f}(1:5398)];
        end   
    end

    disp('-')
    %Features is a feature matrix of 6*channels features
    for finger = 1:5
        pos = dg_subsampled{i}(:, finger);
        N = 3; %time bins before
        M = size(feats,1) - N+1; %Total time bins
        nu = size(feats,2); %number of "neurons" or features
        % 11 features on 62 channels = very inefficient
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
       pos = pos(N+2:end);
            
       [Mdl, FitInfo] = fitrlinear(R, pos, 'Regularization', 'lasso', 'PassLimit', 7)
       %PassLimit 5: testcorr 0.5308
       %PassLimit 7: testcorr 0.5434, 0.5533
       Mdls{i, finger} = Mdl;
       est_pos = predict(Mdl, R);
       x = est_pos(1)*ones(N+2, 1);
       est_pos = [x; est_pos];
       est_pos_full = spline(0:50:270000, est_pos, 0:1:270000);
       predicted_pos{i, finger} = est_pos_full;
       %Lasso
    end
end
%% Do some evaluation
testcorr = 0;
for i = 1:3
    for ch = 1:5
        testcorr  = testcorr + corr(predicted_pos{i,ch}(1:end-1)', dg{i}(:,ch));
    end
end
testcorr/15

%% Do some evaluation
totalcorr = 0;
for i = 1:3
    for ch = 1:5
        totalcorr = totalcorr + corr(predpos_filtfilt{i,ch}, dg{i}(:,ch));
    end
end

totalcorr = totalcorr/15

%% Calculate testing data from f_predictors - testing


%Calculate all metrics again including R matrices

samplingFrequency = 1000;
windowLength = 0.1; %100 ms
overlap = 0.05; %50 ms overlap
windowDisp = windowLength - overlap;

features_leaderboard = cell(3, max(numChannels), 11); %max cell length
%Features = 1 mean, 2-6 the 5 frequency bands

%Functions

%Take average for each sample

avg = @(x) mean(x); %Average of everything in the channel

for i = 1:3 %per subject
    for ch = 1:numChannels(i) %per channel
      features_leaderboard{i, ch, 1} = [MovingWinFeats(ecog_leaderboard{i}(:, ch), 1000, windowLength, windowDisp, avg); 0]; %This seems fake
    end
end

%Evaluate the spectrum at 1000/2 + 1 = 501 frequencies
Fs = 1000;
freqNum = floor(Fs/2) + 1; %We need to have 5-175 Hz - if freqNum is 501, this covers the spectrum

%Frequency bands vary from 0 to 1 pi rad/sample
%Vary from 0 to pi rad/sample * 1000 samples/sec
%Vary from 0 to 1000pi Hz
%0 to 3.1416 Hz

angfreqbands = freqbands*2*pi();
angfreqpercents = angfreqbands/(Fs*pi()); %As a fraction of 1000pi, the max frequency
angfreqindices = floor(angfreqpercents*freqNum);

%Size spectrogram = 501 * 2949

%Find frequency bands for each sample
for i = 1:3 %per channel
    for ch = 1:numChannels(i)
        [spec, f, t] = spectrogram(ecog_leaderboard{i}(:, ch), windowLength*samplingFrequency, overlap*samplingFrequency, Fs);
        for band = 1:size(freqbands, 1)
            features_leaderboard{i, ch, band+1} = abs(mean(spec(angfreqindices(band,:), :)))';

        end
    end
end
%% Make R matrix - testing
predicted_pos_leaderboard = cell(3, 1);
for i = 1:3
    feats = [];
    for ch = 1:numChannels(i)
        for f = 1:size(features, 3)
            norm_features_leaderboard{i,ch,f} = (features_leaderboard{i,ch,f}-means{i,ch,f})/stdevs{i,ch,f};
            sz = 2;
            filt = ones(sz, 1)/sz;
            norm_features_leaderboard{i, ch, f} = conv(norm_features_leaderboard{i, ch, f}, filt, 'same');
            feats = [feats norm_features_leaderboard{i, ch, f}];
        end
    end
    %Features is a feature matrix of 6*channels features
    prediction = [];
    for finger = 1:5
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
        %est_pos = R*f_predictors{i, finger}; %Make predictions
        %x = est_pos(1)*ones(N, 1);
        %est_pos = [x; est_pos];
        %Need to spline it back up to 300,000
        Mdl = Mdls{i,finger};
        est_pos = predict(Mdl, R);
        x = est_pos(1)*ones(N+2, 1);
        est_pos = [x; est_pos];
        %est_pos_full = spline(0:50:270000, est_pos, 0:1:270000);
        %predicted_pos{i, finger} = est_pos_full;
        est_pos_full = spline(0:50:147499, est_pos, 0:1:147499);
        prediction = [prediction est_pos_full'];
    end
    predicted_pos_leaderboard{i} = prediction;
end

%% Do some filtering - testing
sz = 500;
filt = ones(sz, 1)/sz;
predicted_pos_leaderboard_filtered = cell(3,1);
for i = 1:3
    for ch = 1:5
        predicted_pos_leaderboard_filtered{i}(:,ch) = conv(predicted_pos_leaderboard{i}(1:end, ch), filt, 'same');
    end
end
%%

predpos_leaderboard_filtfilt= cell(3,1);
fs = 1000;
band = [0.6 1.3]/(fs/2); %80 - 50 Hz of cutoff
[f, e] = butter(1, band,'stop');
for i = 1:3
    for fing = 1:5
        predpos_leaderboard_filtfilt{i}(:,fing) = filtfilt(f, e, predicted_pos_leaderboard_filtered{i}(:,fing));
    end
end

%% Finalize - testing
predicted_dg = predpos_leaderboard_filtfilt;

