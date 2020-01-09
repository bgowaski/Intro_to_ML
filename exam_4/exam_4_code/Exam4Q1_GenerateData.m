clear all, close all,

plotData = 1;
n = 2; Ntrain = 1000; Ntest = 10000; 
ClassPriors = [0.35,0.65]; r0 = 2; r1 = 4; sigma = 1;

TrainingData.labels = (rand(1,Ntrain) > ClassPriors(1));
ind0train = find(TrainingData.labels==0);ind1train = find(TrainingData.labels==1);
N0 = length(ind0train); N1 = length(ind1train);
theta0 = 2*pi*rand(1,N0); x0 = sigma*randn(n,N0) + r0*[cos(theta0);sin(theta0)];
theta1 = 2*pi*rand(1,N1); x1 = sigma*randn(n,N1) + r1*[cos(theta1);sin(theta1)];
TrainingData.features(:,ind0train) =  x0; TrainingData.features(:,ind1train) =  x1;


TestingData.labels = (rand(1,Ntest) > ClassPriors(1));
ind0test = find(TestingData.labels==0);ind1test = find(TestingData.labels==1);
N0 = length(ind0test); N1 = length(ind1test);
theta0 = 2*pi*rand(1,N0); x0 = sigma*randn(n,N0) + r0*[cos(theta0);sin(theta0)];
theta1 = 2*pi*rand(1,N1); x1 = sigma*randn(n,N1) + r1*[cos(theta1);sin(theta1)];
TestingData.features(:,ind0test) =  x0; TestingData.features(:,ind1test) =  x1;

if plotData == 1
    figure(1), subplot(1,2,1),
    plot(TrainingData.features(1,ind0train),TrainingData.features(2,ind0train),'b.'); hold on, 
    plot(TrainingData.features(1,ind1train),TrainingData.features(2,ind1train),'m.'); axis equal, 
    title('Training Data'),
    subplot(1,2,2),
    plot(TestingData.features(1,ind0test),TestingData.features(2,ind0test),'b.'); hold on, 
    plot(TestingData.features(1,ind1test),TestingData.features(2,ind1test),'m.'); axis equal, 
    title('Testing Data'),
end
%disp(TrainingData.features(1,ind0train));
train1_0 = TrainingData.features(1,ind0train);
train2_0 = TrainingData.features(2,ind0train);
train1_1 = TrainingData.features(1,ind1train);
train2_1 = TrainingData.features(2,ind1train);
test1_0 = TestingData.features(1,ind0train);
test2_0 = TestingData.features(2,ind0train);
test1_1 = TestingData.features(1,ind1train);
test2_1 = TestingData.features(2,ind1train);
%c = {test1_0(:);test1_1(:);test2_0(:);test2_1(:);};
% Convert cell to a table and use first row as variable names
csvwrite('training_label0.csv', [train1_0(:),train2_0(:)]);
csvwrite('training_label1.csv', [train1_1(:),train2_1(:)]);
csvwrite('testing_label0.csv', [test1_0(:),test2_0(:)]);
csvwrite('testing_label1.csv', [test1_1(:),test2_1(:)]);
