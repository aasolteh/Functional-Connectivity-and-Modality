addpath('D:\Amirali\EEGLAB\eeglab_current\eeglab2021.1')
clear; clc
%% Question 1

% 1.1 Pre-Processing Steps
%% EEGLAB
eeglab;

%% Load Saved Data
epochedData = load('data/epochedData.mat');
epochedTime = load('data/epochedTime.mat');
epochedData = epochedData.epochedData;
epochedTime = epochedTime.epochedTime;
epochedData = double(epochedData);

%% 1.2 Representational Dissimilarity Matrices (RDMs)
eventsData = readtable('data/events.csv');

%% Create RDM
objects  = eventsData.object;
idxFace  = find(strcmp(objects, 'face'));
idxChair = find(strcmp(objects, 'chair'));
idxDog   = find(strcmp(objects, 'dog'));

meanTime = find(epochedTime < 225 & epochedTime > 175);

faceEEG  = epochedData(:, meanTime, idxFace);
chairEGG = epochedData(:, meanTime, idxChair);
dogEGG   = epochedData(:, meanTime, idxDog);

meanFace  = mean(mean(faceEEG, 3), 2);
meanChair = mean(mean(chairEGG, 3), 2);
meanDog   = mean(mean(dogEGG, 3), 2);

means  = [meanFace meanChair meanDog];
RDM    = 1 - corr(means, means, 'Type', 'Spearman');
RDMMap = heatmap(RDM);
RDMMap.YDisplayLabels = {'face', 'chair', 'dog'};
RDMMap.XDisplayLabels = {'face', 'chair', 'dog'};

%% RSA model
windowLength = find(epochedTime < min(epochedTime) + 50 & epochedTime >= min(epochedTime));
faceRSA      = nan(1, length(epochedTime) - length(windowLength));
chairRSA     = nan(1, length(epochedTime) - length(windowLength));
dogRSA       = nan(1, length(epochedTime) - length(windowLength));
for i = 1:length(epochedTime) - length(windowLength)
    windowTime = windowLength + i - 1;
    faceEEG    = epochedData(:, windowTime, idxFace);
    chairEGG   = epochedData(:, windowTime, idxChair);
    dogEGG     = epochedData(:, windowTime, idxDog);
    meanFace   = mean(mean(faceEEG, 3), 2);
    meanChair  = mean(mean(chairEGG, 3), 2);
    meanDog    = mean(mean(dogEGG, 3), 2);
    meansFace  = [meanFace meanChair + meanDog];
    meansChair = [meanChair meanFace + meanDog];
    meansDog   = [meanDog meanFace + meanChair];
    faceCorr   = 1 - corr(meansFace, meansFace, 'Type', 'Spearman');
    chairCorr  = 1 - corr(meansChair, meansChair, 'Type', 'Spearman');
    dogCorr    = 1 - corr(meansDog, meansDog, 'Type', 'Spearman');
    faceRSA(i) = faceCorr(2);
    chairRSA(i)= chairCorr(2);
    dogRSA(i)  = dogCorr(2);
end

timeRSA = linspace(1, length(epochedTime) - length(windowLength), length(epochedTime) - length(windowLength));
plot(timeRSA, faceRSA, '--')
hold on
plot(timeRSA, chairRSA, '.-')
hold on
plot(timeRSA, dogRSA)

%% Data Modality
%% Load Data
LFPdata = load('data/lfp.mat').X_in_sessions;
V4data  = LFPdata(1, :, :);
FEFdata = LFPdata(2, :, :);
SR      = 1000;

% clear LFPdata
V4data  = squeeze(V4data);
FEFdata = squeeze(FEFdata);
t = linspace(0, size(FEFdata, 1) / SR, size(FEFdata, 1));

%% 2.1 PLV Analysis
waveBounds = ["delta", "theta", "alpha", "beta"];
waveStart  = [0.1, 4, 8, 16];
waveEnd    = [4, 7, 15, 31];
PLV        = nan(length(waveBounds), length(t));
smoothPLV  = nan(length(waveBounds), length(t));
for i = 1:length(waveBounds)
    bpfilter = designfilt('bandpassfir','FilterOrder',100, ...
             'CutoffFrequency1',waveStart(i),'CutoffFrequency2',waveEnd(i), ...
             'SampleRate', SR);
    V4filtered = filtfilt(bpfilter, V4data);
    FEFfiltered = filtfilt(bpfilter, FEFdata);
    
    V4hilbert = angle(hilbert(V4filtered));
    FEFhilbert = angle(hilbert(FEFfiltered));
    % Calculating Phase Difference (PLV)
    PLV(i, :) = abs(sum(exp(1i * (V4hilbert - FEFhilbert)), 2)) / size(FEFdata, 2);
%     permutationTest(PLV(i, :), ones(1, 4000), 3)
    smoothPLV(i, :) = smoothdata(PLV(i, :));
    subplot(2, 2, i)
    plot(t, smoothPLV(i, :))
    xlabel('time')
    ylabel('PLV')
    title(strcat("PLV for " , waveBounds(i) , " wave"))
end

%% 2.2b Check Preconditions by unit root test
FEFmean = mean(FEFdata, 2)';
V4mean  = mean(V4data, 2)';
plot(t, FEFmean, t, V4mean)
xlabel('time')
ylabel('mean value')
title('Mean value over trial for each time')
legend('FEF', 'V4')
rejectedFEF = 0;
rejectedV4  = 0;
stationaryresultsFEF = nan(size(FEFdata, 2), 1);
stationaryresultsV4  = nan(size(V4data, 2), 1);

for i = 1:size(FEFdata, 2)
    stationaryresultsFEF(i) = kpsstest(FEFdata(:, i));
    stationaryresultsV4(i)  = kpsstest(V4data(:, i));
    nonstationaryFEF        = find(stationaryresultsFEF == 1);
    nonstationaryV4         = find(stationaryresultsV4 == 1);
end

%% 2.2c Preprocessing
%% (i) Filtering
hpfilter = designfilt('highpassfir','FilterOrder',100,'CutoffFrequency',0.5,'StopbandAttenuation',65,'PassbandRipple',0.5,'SampleRate',SR);
fvtool(hpfilter)
V4filtered = filtfilt(hpfilter, V4data);
FEFfiltered = filtfilt(hpfilter, FEFdata);

notchfilter = designfilt('bandstopfir','FilterOrder',100,'CutoffFrequency1',59.9,'CutoffFrequency2',60.1,'PassbandRipple1',2,'StopbandAttenuation',65,'PassbandRipple2',2,'SampleRate',SR);
fvtool(notchfilter)
FEFnotched  = filtfilt(notchfilter, FEFfiltered);
V4notched  = filtfilt(notchfilter, V4filtered);

%% (ii) Normalization
numTrials = size(FEFdata, 2);
FEFnorm   = nan(size(FEFdata));
V4norm    = nan(size(V4data));
for i = 1:numTrials
    FEFnorm(:, i) = (FEFdata(:, i) - mean(FEFdata(:, i))) / std(FEFdata(:, i));
    V4norm(:, i)  = (V4data(:, i) - mean(V4data(:, i))) / std(V4data(:, i));
end

%% (iii) Differentiation
FEFdiff  = diff(FEFnorm, 1, 1);
V4diff   = diff(V4norm, 1, 1);
datadiff = nan(2, size(FEFdiff, 1), size(FEFdata, 2));
datadiff(1, :, :) = FEFdiff;
datadiff(2, :, :) = V4diff;

%% 2.2d LGC method
%% Import data to fieldtrip
cfg = [];
cfg.dataset = 'data/datadiff.set';
ft_data1 = ft_preprocessing(cfg);
ft_data1.label = {'FEF', 'V4'};
%% Plot data to check
plot(ft_data1.time{1}, ft_data1.trial{1})
legend(ft_data1.label)
xlabel('time (s)')

%% Compute frequency of time signals
cfg = [];
cfg.method    = 'mtmfft';
cfg.output    = 'fourier';
cfg.foilim    = [0 200];
cfg.tapsmofrq = 5;
freq          = ft_freqanalysis(cfg, ft_data1);
fd            = ft_freqdescriptives(cfg, freq);

figure;plot(fd.freq, fd.powspctrm);
set(findobj(gcf,'color',[0 0.5 0]), 'color', [1 0 0]);
title('power spectrum');

%% 
% compute connectivity
cfg = [];
cfg.method = 'granger';
g = ft_connectivityanalysis(cfg, freq);
cfg.method = 'coh';
c = ft_connectivityanalysis(cfg, freq);


% visualize the results
cfg = [];
cfg.parameter = 'grangerspctrm';
figure; ft_connectivityplot(cfg, g);
% cfg.parameter = 'cohspctrm';
% figure; ft_connectivityplot(cfg, c);

%% 2.2e Connectivity Strength
figure
plot(g.freq, squeeze(g.grangerspctrm(:,1,:)), g.freq, squeeze(g.grangerspctrm(:,2,:)))
title(sprintf('connectivity between %s and %s', g.label{1}, g.label{2}));
xlabel('freq (Hz)')
ylabel('coherence')
legend(g.label)

%%
grangercfg = [];
grangercfg.method  = 'granger';
grangercfg.granger.conditional = 'no';
grangercfg.granger.sfmethod = 'bivariate';
coherence      = ft_connectivityanalysis(grangercfg, fd.freq);
%% 2.2f 
