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

%% Crossvalidation

%Number of folds
%fold_num=10;
%Calculate number of samples per fold
%n = num_values_train;
%sample_num = n/fold_num;
%folds = cell(1,fold_num);
%Randommly populate the dataset indices in the folds 
%vec=1:fold_num
%for i = 1:fold_num
%    start = num_values_train
%    folds{i}=vec(start:(start+n-1));
%end
%Number of unique sample indices
%disp(length(unique([folds{:}])));

%% Get values
for i = 1:3
    dg{i} = session_dg{i}.data.getvalues(1:270000, 1:5);
    ecog{i} = session_ecog{i}.data.getvalues(1:270000, 1:numChannels(i));
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

%fs = 1000;
%band = [50]/(fs/2); %80 - 50 Hz of cutoff
%[f, e] = butter(2, band);
%for i = 1:3
%    for ch = 1:numChannels(i)
%        ecog_filtered{i}(:,ch) = filtfilt(f, e, ecog{i}(:,ch));
%    end
%end

%ecog = ecog_filtered;

%% Calculate mean features - training
samplingFrequency = 1000;
windowLength = 0.1; %100 ms
overlap = 0.05; %50 ms overlap
windowDisp = windowLength - overlap;

features = cell(3, max(numChannels), 9); %max cell length
%Features = 1 mean, 2-6 the 5 frequency bands

%Functions

%Take average for each sample
avg = @(x) mean(x); %Average of everything in the channel
square = @(x) mean(x.^2) %Square everything in channel
%Line Length function
LLFn = @(x) sum(abs(diff(x)));
%Area function
Area = @(x) sum(abs(x));
for i = 1:3 %per subject
    for ch = 1:numChannels(i) %per channel
      features{i, ch, 1} = [MovingWinFeats(ecog{i}(:, ch), 1000, windowLength, windowDisp, avg); 0]; %This seems fake
      features{i, ch, 2} = [MovingWinFeats(ecog{i}(:, ch), 1000, windowLength, windowDisp, square); 0]; %This seems fake
      features{i, ch, 3} = [MovingWinFeats(ecog{i}(:, ch), 1000, windowLength, windowDisp, LLFn); 0]; %This seems fake
      features{i, ch, 4} = [MovingWinFeats(ecog{i}(:, ch), 1000, windowLength, windowDisp, Area); 0]; %This seems fake
    end
end

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
            features{i, ch, band+4} = abs(mean(spec(angfreqindices(band,:), :)))';
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
featsz=size(features);
numfeats=featsz(3);
means = cell(3, 5, 14);
stdevs = cell(3,5,14);
BETAS = cell(3, 5);

for i = 1:3
    feats = [];
    for ch = 1:numChannels(i)
        for f = 1:numfeats
            means{i,ch,f} = mean(features{i,ch,f});
            stdevs{i,ch,f} = std(features{i,ch,f});
            norm_features{i,ch,f} = (features{i,ch,f}-means{i,ch,f}) / stdevs{i,ch,f};
            sz = 2;
            filt = ones(sz, 1)/sz;
            norm_features{i, ch, f} = conv(norm_features{i, ch, f}, filt, 'same');
            size(norm_features{i,ch,f})
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
           
       [Mdl, FitInfo] = fitrlinear(R, pos, 'Regularization', 'lasso', 'PassLimit', 10, 'Solver', 'asgd', 'KernelFunction', 'gaussian')
       %PassLimit 5: testcorr 0.5308
       %PassLimit 7: testcorr 0.5434, 0.5533 (submitted one)
       %PassLimit 10:testcorr 0.5563 (still iteration limitexceeded)
       %Passlimit 10 and asgd: 0.5620 (still iteration limit exceeded)
       %passlimit 10 and asgd with the extra features: 0.6085, it exceeded
       Mdls{i, finger} = Mdl;
       est_pos = predict(Mdl, R);
       x = est_pos(1)*ones(N+2, 1);
       est_pos = [x; est_pos];
       est_pos_full = spline(0:50:270000, est_pos, 0:1:270000);
       predicted_pos{i, finger} = est_pos_full;
    end
end
%% Do some evaluation
testcorr = 0;
for i = 1:3
    for ch = 1:5
        corr(predicted_pos{i,ch}(1:end-1)', dg{i}(:,ch))
        testcorr  = testcorr + corr(predicted_pos{i,ch}(1:end-1)', dg{i}(:,ch));
    end
end
testcorr/15
%%
disp('-')
sz = 300;
filt = ones(sz, 1)/sz;
filtered_predicted_pos = cell(3,5);
totalcorr = 0;
for i = 1:3
    for ch = 1:5
        filtered_predicted_pos{i, ch} = conv(predicted_pos{i,ch}', filt, 'same');
    end
end

fs = 1000;
band = [0.6 1.5]/(fs/2); %80 - 50 Hz of cutoff
[f, e] = butter(1, band,'stop');
for i = 1:3
    for fing = 1:5
        predpos_filtfilt{i,fing} = filtfilt(f, e, filtered_predicted_pos{i,fing});
    end
end

% high = [0.00005]/(fs/2); %80 - 50 Hz of cutoff
% [f, e] = butter(2, high,'high');
% for i = 1:3
%     for fing = 1:5
%         predpos_fff{i,fing} = filtfilt(f, e, predpos_filtfilt{i,fing});
%     end
% end

%% Do some evaluation
totalcorr = 0;
for i = 1:3
    for ch = 1:5
        corr(predpos_filtfilt{i,ch}(1:end-1), dg{i}(:,ch))
        totalcorr = totalcorr + corr(predpos_filtfilt{i,ch}(1:end-1), dg{i}(:,ch));
    end
end

totalcorr = totalcorr/15

%% Do some evaluation
totalcorr = 0;
for i = 1:3
    for ch = 1:5
        corr(predpos_fff{i,ch}, dg{i}(:,ch))
        totalcorr = totalcorr + corr(predpos_fff{i,ch}, dg{i}(:,ch));
    end
end

totalcorr = totalcorr/15
%% Calculate testing data from f_predictors - testing

%sz = 10;
%filt = ones(sz, 1)/sz;
%for i = 1:3
%    for ch = 1:numChannels(i)
%        ecog_leaderboard{i}(:,ch) = conv(ecog_leaderboard{i}(:, ch), filt, 'same');
%    end
%end

%fs = 1000;
%band = [50]/(fs/2); %80 - 50 Hz of cutoff
%[f, e] = butter(2, band);
%for i = 1:3
%    for ch = 1:numChannels(i)
%        ecog_filtered{i}(:,ch) = filtfilt(f, e, ecog{i}(:,ch));
%    end
%end

%ecog = ecog_filtered;

%Calculate all metrics again including R matrices

samplingFrequency = 1000;
windowLength = 0.1; %100 ms
overlap = 0.05; %50 ms overlap
windowDisp = windowLength - overlap;

features_leaderboard = cell(3, max(numChannels), 6); %max cell length
%Features = 1 mean, 2-6 the 5 frequency bands

%Functions

%Take average for each sample

avg = @(x) mean(x); %Average of everything in the channel

for i = 1:3 %per subject
    for ch = 1:numChannels(i) %per channel
      features_leaderboard{i, ch, 1} = [MovingWinFeats(ecog_leaderboard{i}(:, ch), 1000, windowLength, windowDisp, avg); 0]; %This seems fake
      features_leaderboard{i, ch, 2} = [MovingWinFeats(ecog_leaderboard{i}(:, ch), 1000, windowLength, windowDisp, square); 0]; %This seems fake
      features_leaderboard{i, ch, 3} = [MovingWinFeats(ecog_leaderboard{i}(:, ch), 1000, windowLength, windowDisp, LLFn); 0]; %This seems fake
      features_leaderboard{i, ch, 4} = [MovingWinFeats(ecog_leaderboard{i}(:, ch), 1000, windowLength, windowDisp, Area); 0]; %This seems fake
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
            features_leaderboard{i, ch, band+4} = abs(mean(spec(angfreqindices(band,:), :)))';

        end
    end
end
%% Make R matrix - testing
predicted_pos_leaderboard = cell(3, 1);
for i = 1:3
    feats = [];
    for ch = 1:numChannels(i)
        for f = 1:numfeats
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
sz = 300;
filt = ones(sz, 1)/sz;
predicted_pos_leaderboard_filtered = cell(3,1);
for i = 1:3
    for ch = 1:5
        predicted_pos_leaderboard_filtered{i}(:,ch) = conv(predicted_pos_leaderboard{i}(1:end, ch), filt, 'same');
    end
end

predpos_leaderboard_filtfilt= cell(3,1);
fs = 1000;
band = [0.6 1.3]/(fs/2); %80 - 50 Hz of cutoff
[f, e] = butter(1, band,'stop');
for i = 1:3
    for fing = 1:5
        predpos_leaderboard_filtfilt{i}(:,fing) = filtfilt(f, e, predicted_pos_leaderboard_filtered{i}(:,fing));
    end
end

% high = [0.005]/(fs/2); %80 - 50 Hz of cutoff
% [f, e] = butter(2, high,'high');
% for i = 1:3
%     for fing = 1:5
%         predpos_leaderboard_filtfilt{i}(:,fing) = filtfilt(f, e, predicted_pos_leaderboard_filtered{i}(:,fing));
%     end
% end


%% Finalize - testing
predicted_dg = predpos_leaderboard_filtfilt;


