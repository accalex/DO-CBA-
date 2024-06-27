% function o=getObjValue1(x,label,data)
%     x=round(x);
%     [IW,B,LW,TF,TYPE] = elmtrain(data,label,x,'sig',1);
%     Y=elmpredict(data,IW,B,LW,TF,TYPE);
%     wucha=Y-label;
%     nonzero=length(find(wucha~=0));%找出输出和原始标签不同的元素，算出有几个误差
%     [m,n]=size(label);%获取数组维度
%     o=nonzero/n*100;%误差占比，优化目标最小化
% end
%%
% function o=getObjValue1(x,TrainingData_File,TestingData_File)
%     c=x(1,1);
%     p=x(1,2);
%     k=1;
%     Elm_Type=1;%分类任务，0 for 回归任务
%     Regularization_coefficient=c;%正则化系数
%     switch k        %%核函数选择
%         case 1
%             Kernel_type='RBF_kernel';
%         case 2
%             Kernel_type='lin_kernel';
%         case 3
%             Kernel_type='poly_kernel';
%         case 4
%             Kernel_type='wav_kernel';
%     end
%     Kernel_para=p;
%     [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,TY] = elm_kernel(TrainingData_File, TestingData_File,Elm_Type, Regularization_coefficient, Kernel_type, Kernel_para);
%     o=1-TrainingAccuracy;
% end
%%
% function o=getObjValue1(x,train_data,target1)
%     kk=5;%交叉验证次数
%     [~,n]=size(train_data);
%     indices=crossvalind('kfold',n,kk);
%     TestingAccuracy=0;
%     for k=1:kk
%         test=(indices==k);
%         train=~test;
%         midtrain_data=train_data(:,train);
%         target=target1(train,:);
%         midtest_data=train_data(:,~train);
%         targettest=target1(~train,:);
%         %用非交叉验证的话用上面注释掉的
%         TestingAccuracy_k=BILSTM(x,midtrain_data,target,midtest_data,targettest); 
%         TestingAccuracy=TestingAccuracy+TestingAccuracy_k;
%     end    
% %     o=((1-TrainingAccuracy/kk)+(1-TestingAccuracy/kk))/2;
%       o=1-TestingAccuracy/kk;
% end


function o=getObjValue1(x,train_data,target1)
        accuracy=BILSTM(x,train_data,target1,train_data,target1); 
      o=1-accuracy;
end
