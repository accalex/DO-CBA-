%__________________________________________________________________
%  Dandelion Optimizer
%  Developed in MATLAB R2018a
%
%  programmer: Shijie Zhao and Tianran Zhang
%
%  The code is based on the following papers.
%  Shijie Zhao, Tianran Zhang, Shilin Ma, and Miao Chen
%  Dandelion Optimizer: A nature-inspired metaheuristic algorithm for
%  engineering applications.
%  Engineering Applications of Artificial Intelligence
%  DOI:10.1016/j.engappai.2022.105075
%
%__________________________________________________________________

function [Best_fitness,Best_position,Convergence_curve]=DO(Popsize,Maxiteration,LB,UB,Dim,Fobj)
tic;

dandelions=initialization(Popsize,Dim,UB,LB);%%%%%%%%  initialization
dandelionsFitness = zeros(1,Popsize);
Convergence_curve=zeros(1,Maxiteration);
for i=1:Popsize
    dandelionsFitness(1,i)=Fobj(dandelions(i,:));
end
% Calculate the fitness values of initial dandelions.
[~,sorted_indexes]=sort(dandelionsFitness);
Best_position=dandelions(sorted_indexes(1),:);
Best_fitness = dandelionsFitness(sorted_indexes(1));
Convergence_curve(1)=Best_fitness;
t=2;

while t<Maxiteration+1
    
    %% Rising stage
    beta=randn(Popsize,Dim);
    rand_=randperm(Popsize);
    alpha=rand()*((1/Maxiteration^2)*t^2-2/Maxiteration*t+1); % eq.(8) in this paper
    a=-1/(Maxiteration^2-2*Maxiteration+1);
    b=-2*a;
    c=1-a-b;
    k=1-rand()*(c+a*t^2+b*t); % eq.(11) in this paper
    if randn()<1.5%%%%%%%%%%%%%% 1.5越小越全局
        for i=1:Popsize
            lamb=abs(randn(1,Dim));
            theta=(2*rand()-1)*pi;
            row=1/exp(theta);
            vx=row*cos(theta);
            vy=row*sin(theta);
            NEW=rand(1,Dim).*(UB-LB)+LB;
            dandelions_1(i,:)=dandelions(i,:)+alpha.*vx.*vy.*lognpdf(lamb,0,1).*(NEW(1,:)-dandelions(i,:)); % eq.(5) in this paper
        end
    else
        for i=1:Popsize
            dandelions_1(i,:)=dandelions(i,:).*k; % eq.(10) in this paper
            
        end
    end
    dandelions=dandelions_1;
    % Check boundries
    dandelions = max(dandelions,LB);
    dandelions = min(dandelions,UB);
    
    %% Decline stage
    dandelions_mean=sum(dandelions,1)/Popsize; % eq.(14) in this paper
    for i=1:Popsize
        for j=1:Dim
            dandelions_2(i,j)=dandelions(i,j)-beta(i,j)*alpha*(dandelions_mean(1,j)-beta(i,j)*alpha*dandelions(i,j)); % eq.(13) in this paper
        end
    end
    dandelions=dandelions_2;
    % Check boundries
    dandelions = max(dandelions,LB);
    dandelions = min(dandelions,UB);
    
    %% Landing stage
    Step_length=levy(Popsize,Dim,1.5);
    Elite=repmat(Best_position,Popsize,1);
    for i=1:Popsize
        for j=1:Dim
            dandelions_3(i,j)=Elite(i,j)+Step_length(i,j)*alpha*(Elite(i,j)-dandelions(i,j)*(2*t/Maxiteration)); % eq.(15) in this paper
        end
    end
    dandelions=dandelions_3;
    % Check boundries
    dandelions = max(dandelions,LB);
    dandelions = min(dandelions,UB);
    
    %%
    % Calculated all dandelion seeds' fitness values
    for i=1:Popsize
        dandelionsFitness(1,i)=Fobj(dandelions(i,:));
    end
    
    
    % Arrange dandelion seeds from good to bad according to fitness values
    [~,sorted_indexes]=sort(dandelionsFitness);
    dandelions=dandelions(sorted_indexes(1:Popsize),:);
    SortfitbestN = dandelionsFitness(sorted_indexes(1:Popsize));
    
    %Update the optimal dandelion seed
    if SortfitbestN(1)<Best_fitness
        Best_position=dandelions(1,:);
        Best_fitness=SortfitbestN(1);
    end
    
    Convergence_curve(t)=Best_fitness;
    t = t + 1;
end
time = toc;
end


% ___________________________________
function [z] = levy(n,m,beta)
% beta is set to 1.5 in this paper
num = gamma(1+beta)*sin(pi*beta/2);
den = gamma((1+beta)/2)*beta*2^((beta-1)/2);
sigma_u = (num/den)^(1/beta);
u = random('Normal',0,sigma_u,n,m);
v = random('Normal',0,1,n,m);
z =u./(abs(v).^(1/beta));
end
