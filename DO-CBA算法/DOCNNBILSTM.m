N=5;
Max_iter=50;
lb = [0.001, 0.001, 0.001, 10,10,20,1];%p取0.1-10 0.001,1000
ub = [0.01, 0.01, 0.01, 30,10,50,10];
dim=7;
XTrainReshaped_1=train_data(:,:,:,1:3655);
train_data=XTrainReshaped_1;
target1=targetD(1:3655);
fobj=@(x) getObjValue1([x(1:3), round(x(4:end))],train_data,target1);

[Bestfitness,Bestposition,Convergence_curve]=DO(N,Max_iter,lb,ub,dim,fobj);%蒲公英优化算法

x=[Bestposition(1:3), round(Bestposition(4:end))];
    learning_rate = x(1,1);
    learning_rate1 = x(1,2);
    learning_rate2 = x(1,3);
    hidden_size_1 = x(1,4);
    maxepoch=x(1,5);
    miniBatchSize = x(1,6);
    numofattention=x(1,7);
    
    bilstmLayer1 = bilstmLayer(hidden_size_1, 'OutputMode', 'last');
    bilstmLayer1.InputWeightsLearnRateFactor = learning_rate;
    bilstmLayer1.RecurrentWeightsLearnRateFactor = learning_rate1;
    bilstmLayer1.BiasLearnRateFactor = learning_rate2;
    selfAttentionLayer1 = selfAttentionLayer(5, numofattention*5,"Name",'attentionlayer1');
    layers_1 = [
    imageInputLayer([100 1 1],"Name","imageinput")
    convolution2dLayer([1 3],256,"Name","conv","Padding","same")
    leakyReluLayer(0.01,"Name","relu")
    batchNormalizationLayer("Name","batchnorm")
    maxPooling2dLayer([5 5],"Name","maxpool","Padding","same")
    convolution2dLayer([2 3],256,"Name","conv_1","Padding","same")
    leakyReluLayer(0.01,"Name","relu2")
    batchNormalizationLayer("Name","batchnorm_1")
    convolution2dLayer([1 3],64,"Name","conv_2","Padding","same")
    leakyReluLayer(0.01,"Name","relu2_1")
    batchNormalizationLayer("Name","batchnorm_2")
    convolution2dLayer([1 3],32,"Name","conv_3","Padding","same")
    leakyReluLayer(0.01,"Name","relu3")
    batchNormalizationLayer("Name","batchnorm_3")
    globalMaxPooling2dLayer("Name","gmpool")
    flattenLayer("Name","flatten")
    bilstmLayer1
    selfAttentionLayer1
    dropoutLayer(0.2,"Name","dropout")
    fullyConnectedLayer(2,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
    numSamples = size(train, 4);
% 生成随机索引
randIndices = randperm(numSamples);

% 选择前一部分索引作为验证集的索引
validationIndices = randIndices(1:1500); % 假设选择前1500个作为验证集

% 提取验证集数据
XValidation = train(:, :, :, validationIndices);
targetDValidation = target(validationIndices);

% 提取训练集数据
XTrainPartial = train;
XTrainPartial(:, :, :, validationIndices) = [];
targetDPartial = target;
targetDPartial(validationIndices) = [];

options = trainingOptions('sgdm', ...
    'InitialLearnRate', learning_rate, ...
    'MaxEpochs', maxepoch, ...
    'MiniBatchSize', miniBatchSize, ...
    'ValidationData', {XValidation, targetDValidation}, ...  % 设置验证数据为 XValidation 和 targetDValidation
    'ValidationFrequency', 20, ...
    'Verbose', 0, ...
    'Plots', "training-progress");
CNN_network = trainNetwork(train, target, layers_1, options);

