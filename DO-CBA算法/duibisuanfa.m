desired_snr_dB =-20;

% 遍历每个 cell 进行噪声添加
for i = 1:length(XTrain)
    signal = XTrain{i};
    
    % 计算信号的功率
    signal_power = mean(signal .^ 2);
    
    % 根据所需的 SNR 计算噪声功率
    desired_snr_linear = 10^(desired_snr_dB / 10);
    noise_power = signal_power / desired_snr_linear;
    
    % 生成高斯噪声
    noise = sqrt(noise_power) * randn(size(signal));
    
    % 将噪声添加到信号中
    noisy_signal = signal + noise;
    
    % 更新 cell 数组
    XTrain1{i} = noisy_signal;
end

numSamples = numel(XTrain1); % 获取样本数量

% 初始化一个数组来存储调整后的数据
XTrainReshaped = zeros(100, 1, 1, numSamples);

% 将每个序列重塑为网络所需的格式
for i = 1:numSamples
    sequence = XTrain{i}; % 获取当前序列
    reshapedSequence = reshape(sequence, [100, 1, 1]); % 重塑为100×1×1
    XTrainReshaped(:, :, :, i) = reshapedSequence; % 存储到调整后的数组中
end

numSamples = size(XTrainReshaped, 4); % 获取样本数量

% 生成随机索引
randIndices = randperm(numSamples);

% 选择前一部分索引作为验证集的索引
validationIndices = randIndices(1:1500); % 假设选择前1500个作为验证集

% 提取验证集数据
XValidation = XTrainReshaped(:, :, :, validationIndices);
targetDValidation = targetD(validationIndices);

% 提取训练集数据
XTrainPartial = XTrainReshaped;
XTrainPartial(:, :, :, validationIndices) = [];
targetDPartial = targetD;
targetDPartial(validationIndices) = [];

options1 = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 50, ...
    'InitialLearnRate', 0.01, ...
    'ValidationData', {XValidation, targetDValidation}, ...  % 设置验证数据为 XValidation 和 targetDValidation
    'ValidationFrequency', 20, ...
    'Verbose', 1);
net_CNN2 = trainNetwork(XTrainReshaped, targetD, layers_7, options1);
net_resnet = trainNetwork(XTrainReshaped, targetD, lgraph_1, options1);
net_densenet = trainNetwork(XTrainReshaped, targetD, lgraph_2, options1);
net_CLDNN = trainNetwork(XTrainReshaped, targetD, layers_2, options1);
net_CA = trainNetwork(XTrainReshaped, targetD, layers_3, options1);
net_C = trainNetwork(XTrainReshaped, targetD, layers_4, options1);
net_CB = trainNetwork(XTrainReshaped, targetD, layers_5, options1);
net_CAB = trainNetwork(XTrainReshaped, targetD, layers_6, options1);


signals_predict_class = classify(net_resnet, XTrainReshaped, ...
                            MiniBatchSize=50, ...
                            SequencePaddingDirection="left");
accuracy = sum(signals_predict_class == targetD)/numel(XTrain)