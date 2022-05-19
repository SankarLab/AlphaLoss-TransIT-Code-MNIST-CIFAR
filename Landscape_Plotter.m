clc;
clear;
M = 5000; %number of samples. Want this large enough for LLN to kick in.
%rng('shuffle');
rng('default');
%rng(4);
%load('7.mat')
%3,5
%Definition of the distribution
mu = randn(2,2);
%mu = [-.25 0;-.25 .25];
%mu = [-1 -1; 1 1];
m = 2;
ii = randn(m);
sigma = ii*ii.';
%sigma = [.5 .5; .5 2];%eye(2,2);%cat(3, 1, 1); 
%sigma = [1 0; 0 1];
mix = [.5,.5];
gm = gmdistribution(mu,sigma,mix);
rng('default'); % For reproducibility
[X,compIdx] = random(gm,M);
Y = compIdx - ones(M,1);
%We can add Gaussian noise to make more interesting folds
%r = normrnd(-3,2,[M,2]);
%X = X+r;
%We can also plot the distribution
%ezplot(@(x) pdf(gm,x));
%fsurf(@(x,y)reshape(pdf(gm,[x(:) y(:)]),size(x)),[-10 10])

n = 30; %resolution of parameter
%a = [.9,.975,1,2,4,10];
%a = [.95,1,2,10];
%a = [10,1000000000000000000000000000000];
a = [1,2];
%a = [.9,1];
num_a = max(length(a));
param = 10; %range of theta (radius)


t1 = linspace(-param,param,n);
t2 = linspace(-param,param,n);
o = zeros(n,n,M);

l = zeros(n,n,num_a);

for p = 1:1:num_a
    
    if a(p) == 1
       for i = 1:1:n
            for j = 1:1:n
                for k = 1:1:M
                    o(i,j,k) = (1/M)*(-Y(k,1)*(log(1/(1+exp(-t1(i)*X(k,1)-t2(j)*X(k,2))))) - (1-Y(k,1))*log((1/(1+exp(t1(i)*X(k,1)+t2(j)*X(k,2))))));
                end
                l(i,j,p) = sum(o(i,j,:));
            end
        end
    else
        for i = 1:1:n
            for j = 1:1:n
                for k = 1:1:M
                    o(i,j,k) = (1/M)*(a(p)/(a(p)-1))*(1 - Y(k,1)*(1/(1+exp(-t1(i)*X(k,1)-t2(j)*X(k,2))))^(1-1/a(p)) - (1-Y(k,1))*(1/(1+exp(t1(i)*X(k,1)+t2(j)*X(k,2))))^(1-1/a(p)));
                end
                l(i,j,p) = sum(o(i,j,:));
            end
        end
    end

    %minMatrix(v) = min(min(o(:,:,v)));
    %[row(v),col(v)] = find(o(:,:,v) == minMatrix(v));

end    
figure;

for p = 1:1:num_a
%set(gcf, 'Position',  [100, 100, 500, 400])
subplot(1,2,p)
%subplot(1,2,p)
surf(t1,t2,l(:,:,p))
hold on
% h = zeros(b);
% for v = 1:1:b
%     if (v == 1)
%         h(v) = surf(t1,t2,o(:,:,v));
%         hold on
%         plot3(t1(col(v)),t2(row(v)), minMatrix(v), 'gp', 'MarkerSize', 10, 'MarkerFaceColor','r')
%     else
%         h(v) = surf(t1,t2,o(:,:,v),'FaceColor','r');%'edgecolor','r')%
%         hold on
%         plot3(t1(col(v)),t2(row(v)), minMatrix(v), 'gp', 'MarkerSize', 10, 'MarkerFaceColor','g')
%     end
% end

rad = 15;
zlabel('\alpha-Risk','FontSize',rad)
xlabel('\theta^{1}','FontSize',rad)
ylabel('\theta^{2}','FontSize',rad)
if p == 2
    %title("\alpha = \infty",'FontSize',rad)
    title("\alpha = " + a(p),'FontSize',rad)
else
    title("\alpha = " + a(p),'FontSize',rad)
end

%lgd = legend(h(1:b),'Expected Risk','20 Samples');
%lgd = legend(h(1:b),'Expected Risk');
%lgd.FontSize = rad-5;
%view([137,14])
view([-46 11])
%view([134,13])
%view([131,19])
end