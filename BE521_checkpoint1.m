session_dg = cell(3,1);
session_ecog = cell(3,1);
dg = cell(3, 1);
ecog = cell(3, 1);
ecog_filtered = cell(3, 1);
for i = 1:3
    str_dg = strcat('I521_Sub', num2str(i), '_Training_dg')
    str_ecog = strcat('I521_Sub', num2str(i), '_Training_ecog')
    session_dg{i} = IEEGSession(str_dg, 'jaimiec', '//home/jaimiec/Documents/Spring_2018/BE521/Tutorial/jai_ieeglogin.bin');
    session_ecog{i} = IEEGSession(str_ecog, 'jaimiec', '//home/jaimiec/Documents/Spring_2018/BE521/Tutorial/jai_ieeglogin.bin');
end

%% Ecog sessions
session_ecog_leaderboard = cell(3,1);
ecog_leaderboard = cell(3, 1);

%Get the sessions
for i = 1:3
    str_ecog = strcat('I521_Sub', num2str(i), '_Leaderboard_ecog')
    session_ecog_leaderboard{i} = IEEGSession(str_ecog, 'jaimiec', '//home/jaimiec/Documents/Spring_2018/BE521/Tutorial/jai_ieeglogin.bin');
end

%Get the actual data for each session
numChannels = [62, 48, 64]; %Varies per subject
for i = 1:3
    ecog_leaderboard{i} = session_ecog_leaderboard{i}.data.getvalues(1:147500, 1:numChannels(i));
end

%%
numChannels = [62, 48, 64]; %Varies per subject
for i = 1:3
    dg{i} = session_dg{i}.data.getvalues(1:299999, 1:5);
    ecog{i} = session_ecog{i}.data.getvalues(1:299999, 1:numChannels(i));
end

%% Filter the ecog data

%sz = 10;
%filt = ones(sz, 1)/sz;
%for i = 1:3
%    for ch = 1:numChannels(i)
%        ecog{i}(:,ch) = conv(ecog{i}(:, ch), filt, 'same');
%    end
%end

%% Butterworth filter on the ecog data

%d = designfilt('lowpassfir', 'PassbandFrequency', 190, 'StopbandFrequency', 200, 'PassbandRipple', 1, 'StopbandAttenuation', 60, 'SampleRate', 1000);
%ecog_filtered = ecog;

%for i = 1:3
%    for ch = 1:numChannels(i)
%        ecog_filtered{i}(:,ch) = filtfilt(d, ecog{i}(:,ch));
%    end
%end

fs = 1000;
band = [50]/(fs/2); %80 - 50 Hz of cutoff
[f, e] = butter(2, band);
for i = 1:3
    for ch = 1:numChannels(i)
        ecog_filtered{i}(:,ch) = filtfilt(f, e, ecog{i}(:,ch));
    end
end

ecog = ecog_filtered;

%%
samplingFrequency = 1000;
windowLength = 0.1; %100 ms
overlap = 0.05; %50 ms overlap
windowDisp = windowLength - overlap;

features = cell(3, max(numChannels), 6); %max cell length
%Features = 1 mean, 2-6 the 5 frequency bands

%Functions

%Take average for each sample

avg = @(x) mean(x) %Average of everything in the channel

for i = 1:3 %per subject
    for ch = 1:numChannels(i) %per channel
      features{i, ch, 1} = [MovingWinFeats(ecog{i}(:, ch), 1000, windowLength, windowDisp, avg); 0] %This seems fake
    end
end

%%
%Evaluate the spectrum at 1000/2 + 1 = 501 frequencies
Fs = 1000;
freqNum = floor(Fs/2) + 1; %We need to have 5-175 Hz - if freqNum is 501, this covers the spectrum

%Frequency bands vary from 0 to 1 pi rad/sample
%Vary from 0 to pi rad/sample * 1000 samples/sec
%Vary from 0 to 1000pi Hz
%0 to 3.1416 Hz

freqbands = [5 15; 20 25; 75 115; 125 160; 160 175]
angfreqbands = freqbands*2*pi()
angfreqpercents = angfreqbands/(Fs*pi()) %As a fraction of 1000pi, the max frequency
angfreqindices = floor(angfreqpercents*freqNum)
%Size spectrogram = 501 * 2949

%Find frequency bands for each sample
for i = 1:3 %per channel
    for ch = 1:numChannels(i)
        [spec, f, t] = spectrogram(ecog{i}(:, ch), windowLength*samplingFrequency, overlap*samplingFrequency, Fs);
        for band = 1:5
            features{i, ch, band+1} = abs(mean(spec(angfreqindices(band,:), :)))'
        end
    end
end

%%
dg_subsampled = cell(3, 1);
for i = 1:3
    decimated = [];
    for finger = 1:5
        decimated = [decimated decimate(dg{i}(:, finger), 50)];
    end
    dg_subsampled{i} = decimated;
end

%%
predicted_pos = cell(3, 5);
f_predictors = cell(3, 5);
means = cell(3, 5, 6);
stdevs = cell(3,5,6);
for i = 1:3
    feats = []
    for ch = 1:numChannels(i)
        for f = 1:6
            means{i,ch,f} = mean(features{i,ch,f});
            stdevs{i,ch,f} = std(features{i,ch,f});
            norm_features{i,ch,f} = (features{i,ch,f}-means{i,ch,f}) / stdevs{i,ch,f};
            feats  = [feats norm_features{i, ch, f}];
        end
    end

    %Features is a feature matrix of 6*channels features
    for finger = 1:5
        pos = dg_subsampled{i}(:, finger)
        N = 6; %time bins before
        M = size(feats,1) - N+1; %Total time bins
        nu = size(feats,2) %number of "neurons" or features
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
        f = mldivide((R'*R), (R'*pos));
        f_predictors{i, finger} = f;
        est_pos = R*f;
        x = est_pos(1)*ones(N+1, 1);
        est_pos = [x; est_pos];
        %Need to spline it back up to 300,000
        
        est_pos_full = spline(0:50:299999, est_pos, 0:1:299999);
        predicted_pos{i, finger} = est_pos_full;
    end
end
%% Do some evaluation
for i = 1:3
    for ch = 1:5
        corr(predicted_pos{i,ch}(1:end-1)', dg{i}(:,ch))
    end
end
disp('-')
sz = 200;
filt = ones(sz, 1)/sz;
filtered_predicted_pos = cell(3,5);
totalcorr = 0;
for i = 1:3
    for ch = 1:5
        filtered_predicted_pos{i, ch} = conv(predicted_pos{i,ch}(1:end-1)', filt, 'same');
    end
end

%% Do some evaluation
totalcorr = 0;
for i = 1:3
    for ch = 1:5
        totalcorr = totalcorr + corr(predicted_pos{i,ch}(1:end-1)', dg{i}(:,ch));
    end
end

totalcorr = totalcorr/15

%% Calculate testing data from f_predictors


%sz = 10;
%filt = ones(sz, 1)/sz;
%for i = 1:3
%    for ch = 1:numChannels(i)
%        ecog_leaderboard{i}(:,ch) = conv(ecog_leaderboard{i}(:, ch), filt, 'same');
%    end
%end

fs = 1000;
band = [50]/(fs/2); %80 - 50 Hz of cutoff
[f, e] = butter(2, band);
for i = 1:3
    for ch = 1:numChannels(i)
        ecog_filtered{i}(:,ch) = filtfilt(f, e, ecog{i}(:,ch));
    end
end

ecog = ecog_filtered;

%Calculate all metrics again including R matrices

samplingFrequency = 1000;
windowLength = 0.1; %100 ms
overlap = 0.05; %50 ms overlap
windowDisp = windowLength - overlap;

features_leaderboard = cell(3, max(numChannels), 6); %max cell length
%Features = 1 mean, 2-6 the 5 frequency bands

%Functions

%Take average for each sample

avg = @(x) mean(x) %Average of everything in the channel

for i = 1:3 %per subject
    for ch = 1:numChannels(i) %per channel
      features_leaderboard{i, ch, 1} = [MovingWinFeats(ecog_leaderboard{i}(:, ch), 1000, windowLength, windowDisp, avg); 0] %This seems fake
    end
end

%Evaluate the spectrum at 1000/2 + 1 = 501 frequencies
Fs = 1000;
freqNum = floor(Fs/2) + 1; %We need to have 5-175 Hz - if freqNum is 501, this covers the spectrum

%Frequency bands vary from 0 to 1 pi rad/sample
%Vary from 0 to pi rad/sample * 1000 samples/sec
%Vary from 0 to 1000pi Hz
%0 to 3.1416 Hz

freqbands = [5 15; 20 25; 75 115; 125 160; 160 175]
angfreqbands = freqbands*2*pi()
angfreqpercents = angfreqbands/(Fs*pi()) %As a fraction of 1000pi, the max frequency
angfreqindices = floor(angfreqpercents*freqNum)
%Size spectrogram = 501 * 2949

%Find frequency bands for each sample
for i = 1:3 %per channel
    for ch = 1:numChannels(i)
        [spec, f, t] = spectrogram(ecog_leaderboard{i}(:, ch), windowLength*samplingFrequency, overlap*samplingFrequency, Fs);
        for band = 1:5
            features_leaderboard{i, ch, band+1} = abs(mean(spec(angfreqindices(band,:), :)))'
        end
    end
end
%%
predicted_pos_leaderboard = cell(3, 1);
for i = 1:3
    feats = []
    for ch = 1:numChannels(i)
        for f = 1:6
            norm_features_leaderboard{i,ch,f} = (features_leaderboard{i,ch,f}-means{i,ch,f})/stdevs{i,ch,f};
            feats = [feats norm_features_leaderboard{i, ch, f}];
        end
    end
    %Features is a feature matrix of 6*channels features
    prediction = []
    for finger = 1:5
        N = 6; %time bins before
        M = size(feats,1) - N+1; %Total time bins
        nu = size(feats,2) %number of "neurons" or features
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
        est_pos = R*f_predictors{i, finger}; %Make predictions
        x = est_pos(1)*ones(N, 1);
        est_pos = [x; est_pos];
        %Need to spline it back up to 300,000
        est_pos_full = spline(0:50:147499, est_pos, 0:1:147499);
        prediction = [prediction est_pos_full'];
    end
    predicted_pos_leaderboard{i} = prediction;
end

%% Do some filtering
sz = 1000;
filt = ones(sz, 1)/sz;
predicted_pos_leaderboard_filtered = cell(3,1);
for i = 1:3
    for ch = 1:5
        predicted_pos_leaderboard_filtered{i}(:,ch) = conv(predicted_pos_leaderboard{i}(1:end, ch), filt, 'same');
    end
end

%%
predicted_dg = predicted_pos_leaderboard_filtered;

