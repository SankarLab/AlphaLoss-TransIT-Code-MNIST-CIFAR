clc;
clear;
rng('shuffle')
%plane plotter
%M = 50;
M = 100; %num samples
m = 2;
n_line = 100;
pmix1 = .03;
%pmix1 = .05; %good imbalance probability
pmix = [pmix1, 1-pmix1];
%mu = [-10 -10 ; -.5 .5];
% sigma = [3 .5 ; .5 1];
%mu = [0 0;1 1];
%mu1 = -.5;
mu1 = -1;
mu = [mu1 mu1; -mu1 -mu1];
sigma = [1 0; 0 1];
gm = gmdistribution(mu,sigma,pmix);

a = [.65,.8,1,2.5,4];
num_a = max(length(a));

epsilon = .0001; %optimality parameter

eta = .01*ones(5,1); %learning rates for base experiments

max_counter = 20000;
%num_runs = 101;
num_runs = 2;
run_counter = num_runs;
thetas_super_storage = zeros(3,num_a);
errors = 0;

while run_counter > 1
[x,compIdx] = random(gm,M);
y = compIdx-1;
while sum(y) ~= M*(1-pmix)
    [x,compIdx] = random(gm,M);
    y = compIdx-1;
end

%%Noise code
% for i = 1:1:M
%     if y(i) == 0
%         y(i) = binornd(1,0.2);
%     else
%     end
% end

L = zeros(M,3,num_a);
random_range = 2;
randomrangeinit = randi([-random_range,random_range],1);
while randomrangeinit == 0
     randomrangeinit = randi([-random_range,random_range],1);
end    
theta_start = randomrangeinit*randn(3,1); %randomized starting point

bias = ones(M,1);
x_temp = zeros(M,3);
x_temp(:,1) = x_temp(:,1) + bias;
x_temp(:,2) = x_temp(:,2)+ x(:,1);
x_temp(:,3) = x_temp(:,3)+ x(:,2);

theta_storage = zeros(3,num_a);

for p = 1:1:num_a
    count = 0;
    theta_previous = zeros(3,1);
    theta = theta_start;
    while norm(theta-theta_previous) > epsilon 
        theta_previous = theta;
    
        for j = 1:1:M
            L(j,:,p) = ((1-y(j))*sigmoid(theta,x_temp(j,:))*(1-sigmoid(theta,x_temp(j,:)))^(1-1/a(p)) - y(j)*(sigmoid(theta,x_temp(j,:)))^(1-1/a(p))*(1-sigmoid(theta,x_temp(j,:))))*x_temp(j,:);
        end
        grad = sum(L(:,:,p))/M;
   
        theta = theta - eta(p)*grad';
   
        count = count+1;
        if count > max_counter  %sometimes it oscillates forever
            break
        end
    end
    disp(count)
    if sum(isnan(theta))>0
        theta_storage(:,p) = theta_storage(:,p);
        disp('NaN-ed')
        errors = errors + 1;
    else    
        theta_storage(:,p) = theta;
        score = 0;
        for i = 1:1:M
            y_hat = round(sigmoid(theta,x_temp(i,:)));
            if y_hat == y(i)
                score = score + 1;
            else
            end
        end
        score = score/M;
        disp(score)
    end
    thetas_super_storage(:,p) = thetas_super_storage(:,p) + theta_storage(:,p);
end
run_counter = run_counter - 1;
disp(run_counter)
end

thetas_super_storage = thetas_super_storage/(num_runs-1);


%%%% Plotting

num_ones = sum(y);
x_ones = zeros(num_ones,2);
x_zeros = zeros(M - num_ones,2);
count_1 = 1;
count_2 = 1;
for i = 1:1:M
    if y(i) == 1
        x_ones(count_1,:) = x(i,:);
        count_1 = count_1 + 1;
    else
        x_zeros(count_2,:) = x(i,:);
        count_2 = count_2 + 1;
    end
end
dot_width = 75;
figure;
scatter(x_ones(:,1),x_ones(:,2),dot_width,'r','filled','DisplayName','Class +1')
hold on
scatter(x_zeros(:,1),x_zeros(:,2),dot_width,'b','filled','DisplayName','Class -1')
hold on

range = 4;
x1 = linspace(-range,range,n_line);
x2 = zeros(1,n_line);

useful1 = dot(mu(1,:),mu(1,:));
useful2 = dot(mu(2,:),mu(2,:));
useful3 = mu(2,:) - mu(1,:);


% for i = 1:1:n_line
%     x2(i) = (1/useful3(2))*(.5*(useful2 - useful1 + 2*log(pmix(1)/pmix(2))) - useful3(1)*x1(i));
% end    
% plot(x1,x2,'k','LineWidth',5)


%%%%%%%% Scoring

%str = "Mixing Distribution " + pmix;
dim = [.15 .2 .3 .3];
%annotation('textbox',dim,'String',"P[Y=-1] = " + pmix1,'FitBoxToText','on');

% str = {'Mean Vector ', + mu};
% annotation('textbox',dim,'String'," " + mu,'FitBoxToText','on');
% 
n_test = 1000000;
pmix_new = [.5, .5];
gm_new = gmdistribution(mu,sigma,pmix_new);


for i = 1:1:n_line
    x2(i) = (1/useful3(2))*(.5*(useful2 - useful1 + 2*log(pmix_new(1)/pmix_new(2))) - useful3(1)*x1(i));
end    
plot(x1,x2,'k','LineWidth',10) %5

x3 = zeros(1,n_line);
for p = 1:1:num_a
    for i = 1:1:n_line
        x3(i) = -(thetas_super_storage(2,p)/thetas_super_storage(3,p))*x1(i)-thetas_super_storage(1,p)/thetas_super_storage(3,p);
    end
    plot(x1,x3,'LineWidth',5) %2.5
    hold on
end
% 
rad = 30;
%rad1 = 20;
%lgd = legend('+1 Class','-1 Class','Bayes Optimal','\alpha = .95','\alpha = 1','\alpha = 2','\alpha = \infty','Location','southeast');
lgd = legend('+1 Class','-1 Class','Bayes Optimal [Balanced]','\alpha = .65','\alpha = .8','\alpha = 1','\alpha = 2.5','\alpha = 4','Location','southeast');
lgd.FontSize = rad-5;
xlabel('X_{1}','FontSize',rad)
ylabel('X_{2}','FontSize',rad)
title("Class Imbalance",'FontSize',rad)
%title("Class Balance",'FontSize',rad)
%title("Class Noise",'FontSize',rad)
ylim([-3, 3])
xlim([-3, 3])
ax = gca;
ax.YAxis.FontSize = rad-5;
ax.XAxis.FontSize = rad-5;
disp('Most Important Scores')
% 
[x_new,compIdx] = random(gm_new,n_test);
y_new = compIdx-1;
y_storage_Bayes = zeros(n_test,1);

for i = 1:1:n_test
    if sign(mu1) == -1
        if x_new(i,2) >= (1/useful3(2))*(.5*(useful2 - useful1 + 2*log(pmix_new(1)/pmix_new(2))) - useful3(1)*x_new(i,1))
        y_storage_Bayes(i) = 1;
        else
        end
    else
        if x_new(i,2) < (1/useful3(2))*(.5*(useful2 - useful1 + 2*log(pmix_new(1)/pmix_new(2))) - useful3(1)*x_new(i,1))
        y_storage_Bayes(i) = 1;
        else
        end
    end
end
score_Bayes = 1-sum(mod(y_new+y_storage_Bayes,2))/n_test;
disp(score_Bayes)
dim = [.7 .5 .4 .4];
%annotation('textbox',dim,'String',"Bayes(Acc): " + score_Bayes,'FitBoxToText','on');

y_storage_alphas = zeros(n_test,1,num_a);

for p = 1:1:num_a
    for i = 1:1:n_test
        if sign(mu1) == -1
            if x_new(i,2) >= -(thetas_super_storage(2,p)/thetas_super_storage(3,p))*x_new(i,1)-thetas_super_storage(1,p)/thetas_super_storage(3,p)
                y_storage_alphas(i,1,p) = 1;
            else
            end
        else
            if x_new(i,2) < -(thetas_super_storage(2,p)/thetas_super_storage(3,p))*x_new(i,1)-thetas_super_storage(1,p)/thetas_super_storage(3,p)
                y_storage_alphas(i,1,p) = 1;
            else
            end
        end
    end
end
score_alphas = zeros(num_a,1);
for p = 1:1:num_a
    score_alphas(p,1) = 1-sum(mod(y_new+y_storage_alphas(:,1,p),2))/n_test;
    disp(score_alphas(p,1))
    temp = .4-.1*p+.1;
    dim = [.75 temp .4 .4];
   % annotation('textbox',dim,'String',score_alphas(p,1),'FitBoxToText','on');
end
% dim = [.7 .4 .4 .4];
% annotation('textbox',dim,'String',"\alpha = 1: " + score_alphas(1,1),'FitBoxToText','on');

disp('Errors')
disp(errors)
%%%%%%%%%%%

% y_new = ~y_new;
% y_storage_Bayes = ~y_storage_Bayes;
% y_storage_alphas = ~y_storage_alphas;

confusion_matrix_Bayes = confusionmat(y_new,y_storage_Bayes);
F1_Bayes = confusion_matrix_Bayes(1,1)/(confusion_matrix_Bayes(1,1) + .5*(confusion_matrix_Bayes(2,1)+confusion_matrix_Bayes(1,2)));
MCC_Bayes = (confusion_matrix_Bayes(1,1)*confusion_matrix_Bayes(2,2) - confusion_matrix_Bayes(2,1)*confusion_matrix_Bayes(1,2))/sqrt((confusion_matrix_Bayes(1,1)+confusion_matrix_Bayes(2,1))*(confusion_matrix_Bayes(1,1)+confusion_matrix_Bayes(1,2))*(confusion_matrix_Bayes(2,2)+confusion_matrix_Bayes(1,2))*(confusion_matrix_Bayes(2,2)+confusion_matrix_Bayes(2,1)));

dim = [.25 .5 .4 .4];
%annotation('textbox',dim,'String',"Bayes(MCC): " + MCC_Bayes,'FitBoxToText','on');
dim1 = [.3 0 .4 .4];
%annotation('textbox',dim1,'String',"Bayes(F1): " + F1_Bayes,'FitBoxToText','on');
dim2 = [.8 .5 .4 .4];
%annotation('textbox',dim2,'String',"Bayes(PFA): " + confusion_matrix_Bayes(1,2)/n_test,'FitBoxToText','on');

confusion_matrix_alphas = zeros(2,2,num_a);
MCC_alphas = zeros(num_a,1);
F1_alphas = zeros(num_a,1);
for p = 1:1:num_a
    confusion_matrix_alphas(:,:,p) = confusionmat(y_new,y_storage_alphas(:,1,p));
    MCC_alphas(p) = (confusion_matrix_alphas(1,1,p)*confusion_matrix_alphas(2,2,p) - confusion_matrix_alphas(1,2,p)*confusion_matrix_alphas(2,1,p))/sqrt((confusion_matrix_alphas(1,1,p)+confusion_matrix_alphas(2,1,p))*(confusion_matrix_alphas(1,1,p)+confusion_matrix_alphas(1,2,p))*(confusion_matrix_alphas(2,2,p)+confusion_matrix_alphas(1,2,p))*(confusion_matrix_alphas(2,2,p)+confusion_matrix_alphas(2,1,p)));
    F1_alphas(p) = confusion_matrix_alphas(1,1,p)/(confusion_matrix_alphas(1,1,p) + .5*(confusion_matrix_alphas(2,1,p)+confusion_matrix_alphas(1,2,p)));
    temp = .4-.1*p+.1;
    dim = [.25 temp .4 .4];
   % annotation('textbox',dim,'String',MCC_alphas(p),'FitBoxToText','on');
    temp1 = .3+.075*p;
    dim1 = [temp1 0 .4 .4];
   % annotation('textbox',dim1,'String',F1_alphas(p),'FitBoxToText','on');
    dim2 = [.8 temp .4 .4];
   % annotation('textbox',dim2,'String',confusion_matrix_alphas(1,2,p)/n_test,'FitBoxToText','on');
end
disp('MCC Scores')
disp(MCC_Bayes)
disp(MCC_alphas)

%
function z = sigmoid(x,y)
temp = dot(x,y);
z = 1/(1+exp(-temp));
end