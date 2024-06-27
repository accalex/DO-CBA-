numHiddenUnits1 = 300;
miniBatchSize = 50;
MinLength=100;


selfAttentionLayer1 = selfAttentionLayer(4, 24,"Name",'attentionlayer1');

layers_1 = [
    sequenceInputLayer(1,"Name","sequenceinput","MinLength",100)
    convolution1dLayer(3,16,"Name","conv1d","Padding","same")
    leakyReluLayer("Name","relu")
    batchNormalizationLayer("Name","bn1")
    maxPooling1dLayer(2,"Name","maxpool1","Stride",2)
    convolution1dLayer(3,32,"Name","conv2","Padding","same")
    batchNormalizationLayer("Name","bn2")
    leakyReluLayer(0.01,"Name","relu2")
    maxPooling1dLayer(2,"Name","maxpool2","Stride",2)
    convolution1dLayer(3,32,"Name","conv3","Padding","same")
    batchNormalizationLayer("Name","bn3")
    leakyReluLayer(0.01,"Name","relu3")
    maxPooling1dLayer(2,"Name","maxpool3","Stride",2)
    globalMaxPooling1dLayer("Name","globalmaxpool1d")
    selfAttentionLayer1
    % bilstmLayer1
    bilstmLayer(64,"Name","biLSTM","OutputMode","last")
    dropoutLayer(0.5,"Name","dropout")
    fullyConnectedLayer(2,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];


% options = trainingOptions(...
%     "sgdm", ... 
%     'InitialLearnRate',0.01,...
%     'LearnRateSchedule','piecewise',...
%     'LearnRateDropFactor',0.1,... 
%     'LearnRateDropPeriod',10,...  
%     'MaxEpochs', 30, ... 
%     'MiniBatchsize', miniBatchSize, ...
%     'Shuffle','every-epoch', ...
%     'GradientThreshold',1, ...
%     ValidationFrequency = 5,...
%     Verbose=0,...
%     Plots="training-progress");


% 生成随机排列的索引

% rng(1); % 设置随机数种子以确保结果可重复
% randIndices = randperm(numel(XTrain));

randIndices=(1:1500);
% 选择前600个索引作为验证集的索引
validationIndices = randIndices(1:1500);

% 从训练数据和目标中选择验证集
XValidation = XTrain(validationIndices);
targetDValidation = targetD(validationIndices);

% 从训练数据和目标中移除验证集
XTrainPartial = XTrain;
XTrainPartial(validationIndices) = [];
targetDPartial = targetD;
targetDPartial(validationIndices) = [];

% 现在，XTrainPartial 和 targetDPartial 可以作为训练数据
% XValidation 和 targetDValidation 可以作为验证数据

options = trainingOptions(...
    "adam", ... 
    'InitialLearnRate',0.01,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,... 
    'LearnRateDropPeriod',10,...  
    'MaxEpochs', 10, ... 
    'MiniBatchsize', miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'GradientThreshold',1, ...
    'ValidationData', {XValidation, targetDValidation}, ... % 更新为新的验证数据集
    'ValidationFrequency', 20, ... % 每5个迭代周期进行一次验证
    'Verbose', 0, ...
    'Plots', "training-progress");


