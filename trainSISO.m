close all,clear all, clc,
tic

%Load data from directory
load 'importRAW_nan60b'
weighttable = compilationS3; % load table
rzttable = compilationS5; % load table
plt=0;

% Pre-processing parameters
Ndtset = 4; % Number data sets
% Savitzky Golay Filter
o = 2; % Order of curve
fl = 17; % frame length or odd number

% Neural Networks Training Parameters
% Using Fixed ID and FD Based on smallest subset to Find Optimal Number of Hidden Node
ID = 1:2; % Input delay
FD = 1:2; % Feed-back delay
NID = length(ID); % Length of input time step delay
NFD = length(FD); % Length of feed-back time step delay
LDB = max([ID,FD]); % Length of the delay buffer = 5
Ndtrain = 0.75; % Ratio data used for training
Ndval = 0.25; % Ratio data used for validation during training
Ndtest = 0; % Ratio data used for test. The test data set is using pattern 5
Hmax = 20; % Maximum number of hidden layer
dH =1;
Hmin = 0; % Minimum number of hidden layer
Ntrials = 10; % Repetition training of each hidden layer
rng('default') % Default random number for reproduction
trainFcn = 'trainlm'; % Training optimization algorithm
NEpochs = 1000; % Maxium number of training iteration
NMaxFail = 6; % Validation check for early stop training process (Train LM)
MUdec = 0.01; % MU decreased value
MUinc = 10; % MU increased value
MUinit = 10; % Initial MU value

% PREPROCESSING DATA
% Convert Table to Array
for n=1:15
    weight_pp(:,n) = table2array(weighttable(:,n+1));
end
for n=1:5
    rzt_pp(:,n) = table2array(rzttable(:,n+1));
end

% Detect and Eliminate Outliers Using Hampel Identifier
k = 100; % number of neighbors
nsigma = 3; % sigma deviation
for n=1:15
    weight_pp(:,n) = hampel(weight_pp(:,n),k,nsigma);
end
for n=1:5
    rzt_pp(:,n) = hampel(rzt_pp(:,n),k,nsigma);
end

% Find Missing value due to outliers or corrupt data (NaN)
% Fill gaps of the missing value using autoregressive model
for n=1:15
    weight_pp(:,n) = fillgaps(weight_pp(:,n));
end

%Convert Array to Table
weight_tab(:,1) = (weighttable(:,1));
for n=1:15
    weight_tab(:,n+1) = array2table(weight_pp(:,n));
end

rzt_tab(:,1) = (rzttable(:,1));
for n=1:5
    rzt_tab(:,n+1) = array2table(rzt_pp(:,n));
end

%Convert Table to Timetable
weight_tt = table2timetable(weight_tab);
rzt_tt = table2timetable(rzt_tab);

% Aggregating data and calculate daily mean
weight_day_tt = retime(weight_tt,'daily','mean');
rzt_day_tt = retime(rzt_tt,'daily','mean');

% Convert Timetable to Table
weight_day_tab = timetable2table(weight_day_tt);
rzt_day_tab = timetable2table(rzt_day_tt);

% Convert Table to Array
for n=1:15
    weight_day(:,n) = table2array(weight_day_tab(:,n+1));
end
for n=1:5
    rzt_day(:,n) = table2array(rzt_day_tab(:,n+1));
end

% Determine size of Array RZT
[ N I ] = size(rzt_day);
len = N-1;
for m=1:len
    for n=1:5
        rzt(m,n) = rzt_day(m,n);
    end
end
rzt(55:60,5) = NaN;
rzt = rzt';

% Savitzky Golay Filter
for n=1:12
    weight_sday(:,n) = sgolayfilt(weight_day(:,n),o,fl);
end
% Because data for plant number 13 until 15 is only 55 days
for n=13:15
    weight_sday(1:55,n) = sgolayfilt(weight_day(1:55,n),o,fl);
end

% Average of 3 Sample
weight_pat(:,1) = (weight_sday(:,1)+weight_sday(:,2)+weight_sday(:,3))/3;
weight_pat(:,2) = (weight_sday(:,4)+weight_sday(:,5)+weight_sday(:,6))/3;
weight_pat(:,3) = (weight_sday(:,7)+weight_sday(:,8)+weight_sday(:,9))/3;
weight_pat(:,4) = (weight_sday(:,10)+weight_sday(:,11)+weight_sday(:,12))/3;
weight_pat(:,5) = (weight_sday(:,13)+weight_sday(:,14)+weight_sday(:,15))/3;

% Growth rate of plant weight
for n=1:5
    weight_rate(:,n) = diff(weight_pat(:,n));
end

% To avoid error in data plotting due to the lack of values
weight_pat(55:60,5) = NaN;
weight_rate(55:60,5) = NaN;

% Transpose matrix
weight_srate = weight_rate';

% Plot Input - Output data for training
figure(1)
subplot(3,1,1);
plot(weight_pat,'DisplayName','weight_day')
ylabel('Growth of plant weight (g)')
legend ('Pattern 1', 'Pattern 2', 'Pattern 3', 'Pattern 4', 'Pattern 5')
xlim([0 60])
subplot(3,1,2);
plot(weight_rate,'DisplayName','weight_rate')
ylabel('Growth rate of plant weight (g day^{-1})')
xlim([0 60])
subplot(3,1,3);
plot(rzt_day,'DisplayName','rzt_day')
ylabel('Root zone temperature (C^{o})')
xlabel('Days after transplanting (d)')
xlim([0 60])

% Convert numeric to cell type data
in1 = num2cell(rzt(1,:));
in2 = num2cell(rzt(2,:));
in3 = num2cell(rzt(3,:));
in4 = num2cell(rzt(4,:));
in5 = num2cell(rzt(5,:));

tar1 = num2cell(weight_srate(1,:));
tar2 = num2cell(weight_srate(2,:));
tar3 = num2cell(weight_srate(3,:));
tar4 = num2cell(weight_srate(4,:));
tar5 = num2cell(weight_srate(5,:));

% Combine 4 into 1 data sets for training process
inputs = catsamples(in1,in2,in3,in4,'pad');
targets = catsamples (tar1,tar2,tar3,tar4,'pad');

% For multiple variable
% inputs[1,:] = catsamples(in1,in2,in3,in4,'pad');
% inputs[2,:] = catsamples(in1,in2,in3,in4,'pad');

% Define data set #5 for independent model test
valin = in5;
valtar = tar5;

X = inputs;
T = targets;

[ I, N ] = size(X);
[ O, N ] = size(T);

% TRAINING PROCESS
Ntrneq = ceil(N*Ndtrain); % Product of element
Hub = (Ntrneq-O)/(NID*I+NFD*O+O+1); % 5.75
NSim = Ntrneq+NFD; % Number of simulation time steps

j=0;
for h = Hmin:dH:Hmax
      j=j+1;
      if h==0
          neto = narxnet(ID,FD,[],'open',trainFcn);
          Nw = (NID*I+NFD*O+1)*O;
      else
          neto = narxnet(ID,FD,h,'open',trainFcn);
          Nw = (NID*I+NFD*O+1)*h+(h+1)*O;
      end
      Ndof = Ntrneq-Nw;
      neto.divideFcn = 'divideblock'; %
      neto.performFcn = 'mse'; % This is default since 'msereg' is obsolete. Set to 'msereg' but revert back automatically to 'mse'
      neto.divideParam.trainRatio = Ndtrain;
      neto.divideParam.valRatio = Ndval;
      neto.divideParam.testRatio = Ndtest;
      neto.trainParam.epochs = NEpochs;
      neto.trainParam.max_fail = NMaxFail;
      neto.trainParam.mu_dec = MUdec;
      neto.trainParam.mu_inc = MUinc;
      neto.trainParam.mu = MUinit;

      [Xo, Xoi, Aoi, To] = preparets(neto,X,{},T); % Use training data to compute training parameters
      for i= 1:Ntrials
              % Save state of RNG for duplication
              so(i,j)           = rng;
              neto             = configure(neto,Xo,To);
              [neto, tro, Yo, Eo, Xof, Aof ] = train(neto,Xo, To, Xoi, Aoi);
              [Xvo, Xvio, Avio, Tvo] = preparets(neto,valin(1:NSim),{},valtar(1:NSim));
              Yvalo             = neto(Xvo,Xvio,Avio);
              Evo               = gsubtract(Tvo,Yvalo);
              stopcrito{i,j}    = tro.stop;
              bestepocho(i,j)   = tro.best_epoch;
              mse_traino(i,j)   = mse(Eo);
              mse_valo(i,j)     = mse(Evo);
              R2valo(i,j)   = regression(Tvo,Yvalo);
              R2traino (i,j)= regression(To,Yo);

              netc = closeloop(neto);
              [Xc, Xci, Aci, Tc] = preparets(netc,X,{},T);
              [netc, trc, Yc, Ec{i,j}, Xcf, Acf ] = train(netc,Xc, Tc, Xci, Aci);
              [Xvc, Xvic, Avic, Tvc] = preparets(netc,valin(1:NSim),{},valtar(1:NSim));
              Yvalc {i,j}       = netc(Xvc,Xvic,Avic);
              Evc{i,j}          = gsubtract(Tvc,Yvalc{i,j});
              stopcritc{i,j}    = trc.stop;
              bestepochc(i,j)   = trc.best_epoch;
              mse_trainc(i,j)   = mse(Ec{i,j});
              mse_valc(i,j)     = mse(Evc{i,j});
              rmse_valc(i,j)    = sqrt(mse_valc(i,j));
              R2valc(i,j)       = regression(Tvc,Yvalc{i,j});
              R2trainc (i,j)    = regression(Tc,Yc);

              [Xvt, Xvit, Avit, Tvt] = preparets(netc,valin(1:45),{},valtar(1:45));
              s20(1:7) = 20;
              s21(1:7) = 21;
              s22(1:7) = 22;
              s23(1:7) = 23;
              s24(1:7) = 24;
              s25(1:7) = 25;
              s26(1:7) = 26;
              s27(1:7) = 27;
              s28(1:7) = 28;
              s29(1:7) = 29;
              s30(1:7) = 30;
              s20(8:NSim) = 20;
              s21(8:NSim) = 21;
              s22(8:NSim) = 22;
              s23(8:NSim) = 23;
              s24(8:NSim) = 24;
              s25(8:NSim) = 25;
              s26(8:NSim) = 26;
              s27(8:NSim) = 27;
              s28(8:NSim) = 28;
              s29(8:NSim) = 29;
              s30(8:NSim) = 30;
              S20 = num2cell(s20);
              S21 = num2cell(s21);
              S22 = num2cell(s22);
              S23 = num2cell(s23);
              S24 = num2cell(s24);
              S25 = num2cell(s25);
              S26 = num2cell(s26);
              S27 = num2cell(s27);
              S28 = num2cell(s28);
              S29 = num2cell(s29);
              S30 = num2cell(s30);
              Y20{i,j} = netc(S20,Xvic,Avic);
              Y21{i,j} = netc(S21,Xvic,Avic);
              Y22{i,j} = netc(S22,Xvic,Avic);
              Y23{i,j} = netc(S23,Xvic,Avic);
              Y24{i,j} = netc(S24,Xvic,Avic);
              Y25{i,j} = netc(S25,Xvic,Avic);
              Y26{i,j} = netc(S26,Xvic,Avic);
              Y27{i,j} = netc(S27,Xvic,Avic);
              Y28{i,j} = netc(S28,Xvic,Avic);
              Y29{i,j} = netc(S29,Xvic,Avic);
              Y30{i,j} = netc(S30,Xvic,Avic);
              avrg20(i,j) = mean(cell2mat(Y20{i,j}));
              avrg21(i,j) = mean(cell2mat(Y21{i,j}));
              avrg22(i,j) = mean(cell2mat(Y22{i,j}));
              avrg23(i,j) = mean(cell2mat(Y23{i,j}));
              avrg24(i,j) = mean(cell2mat(Y24{i,j}));
              avrg25(i,j) = mean(cell2mat(Y25{i,j}));
              avrg26(i,j) = mean(cell2mat(Y26{i,j}));
              avrg27(i,j) = mean(cell2mat(Y27{i,j}));
              avrg28(i,j) = mean(cell2mat(Y28{i,j}));
              avrg29(i,j) = mean(cell2mat(Y29{i,j}));
              avrg30(i,j) = mean(cell2mat(Y30{i,j}));
      end
end

% Find the minimum value of RMSE
[best_RMSE, idx] = min(rmse_valc(:));
[row,col] = ind2sub(size(rmse_valc),idx);

% Find the R2 value of the minimum RMSE
best_R2 = R2valc(row,col);

% Plot Observation vs Simulation Data
typlot(1,:) = cell2mat(Tvc);
typlot(2,:) = cell2mat(Yvalc{row,col});
typlot = typlot';
xplot = cell2mat(Xvc);
xplot = xplot';

figure (2)
subplot(2,1,1)
plot(typlot,'DisplayName','Simulation Output')
ylabel('Growth rate in plant weight (g day^{-1})')
legend ('Observed', 'Simulated')
xlim([0 NSim])
ylim([0 20])
txt = {['RMSE: ' num2str(best_RMSE) ' g'],['MSE: ' num2str(mse_valc(row,col)) ' g'],['R^{2}: ' num2str(R2valc(row,col))]};
text(38,5,txt,'FontSize',9);

subplot(2,1,2)
plot(xplot,'DisplayName','Simulation Output')
xlim([0 NSim])
ylim([15 40])
ylabel('Root zone temperature (^{o}C)')
xlabel('Time step (days)')

% Plot Input-Output Simulation Data
ystat(1,:) = cell2mat(Y20{row,col});
ystat(2,:) = cell2mat(Y21{row,col});
ystat(3,:) = cell2mat(Y22{row,col});
ystat(4,:) = cell2mat(Y23{row,col});
ystat(5,:) = cell2mat(Y24{row,col});
ystat(6,:) = cell2mat(Y25{row,col});
ystat(7,:) = cell2mat(Y26{row,col});
ystat(8,:) = cell2mat(Y27{row,col});
ystat(9,:) = cell2mat(Y28{row,col});
ystat(10,:) = cell2mat(Y29{row,col});
ystat(11,:) = cell2mat(Y30{row,col});
ystat = ystat';

xstat(1,:) = s20;
xstat(2,:) = s21;
xstat(3,:) = s22;
xstat(4,:) = s23;
xstat(5,:) = s24;
xstat(6,:) = s25;
xstat(7,:) = s26;
xstat(8,:) = s27;
xstat(9,:) = s28;
xstat(10,:) = s29;
xstat(11,:) = s30;
xstat = xstat';

figure(3)
subplot(2,1,1);
plot(ystat,'DisplayName','Simulation Output')
legend ('18-20^{o}C','18-21^{o}C','18-22^{o}C','18-23^{o}C','18-24^{o}C','18-25^{o}C','18-26^{o}C','18-27^{o}C','18-28^{o}C','18-29^{o}C','18-30^{o}C')
xlim([0 NSim])
ylabel('Growth rate of plant weight (g day^{-1})')

subplot(2,1,2);
plot(xstat,'DisplayName','Simulation Input')
%legend ('T18','T20','T22','T24','T26','T28','T30')
ylim([15 35])
xlim([0 NSim])
ylabel('Root zone temperature (^{o}C)')
xlabel('Time step (days)')

% Static Relationship
xsim = [20 21 22 23 24 25 26 27 28 29 30];
ysim = [avrg20(row,col),avrg21(row,col),avrg22(row,col),avrg23(row,col),avrg24(row,col),avrg25(row,col),avrg26(row,col),avrg27(row,col),avrg28(row,col),avrg29(row,col),avrg30(row,col)];

% Plot the Relationship
psim = polyfit(xsim, ysim, 2);
fsim = polyval(psim,xsim);
figure(4)
plot(xsim,ysim,'o',xsim,fsim,'-')
legend ('data', 'Fitted line')
ylabel('Average of growth rate of plant weight (g day^{-1})')
xlabel('Root zone temperature (^{o}C)')

% Plot RMSE in Hidden Layer
rmse_hl = min(rmse_valc);
figure(5)
plot(rmse_hl)
ylabel('RMSE (g)')
xlabel('Number of hidden layer')

% Plot Cross-Correlation of Inputs to Error
figure(6)
plotinerrcorr(Xvc,Evc{row,col})

% Plot Auto-Correlation of Error
figure(7)
ploterrcorr(Evc{row,col})

% result = [ (Hmin:dH:Hmax); R2o ]
bestepocho
R2traino
mse_traino
R2valo
mse_valo
stopcrito
bestepochc
R2trainc
mse_trainc
R2valc
mse_valc
rmse_valc
stopcritc
avrg20
avrg22
avrg24
avrg26
avrg28
avrg30
best_RMSE
best_R2
toc
