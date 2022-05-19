clc;
clear;
M = 15000; %number of samples. Want this large enough for LLN to kick in.
m = 2;
rng('shuffle');
%rng('default');
%load('7.mat')
%Definition of random distributions
% mu = randn(2,2);
% m = 2;
% ii = randn(m);
% sigma = ii*ii.';
% mix = [.5,.5];
%gm = gmdistribution(mu,sigma,mix);
%Definition of distribution in paper
mu = [-.18 1.49; -.01 0.16];
pmix = .12;
sigma1 = [3.2 -2.02; -2.02 2.71];
sigma2 = [4.19 1.27; 1.27 0.9];
sigma = zeros(m,m,2);
sigma(:,:,1) = sigma1;
sigma(:,:,2) = sigma2;
gm = gmdistribution(mu,sigma,[pmix,1-pmix]);

[X,compIdx] = random(gm,M);
Y = compIdx - ones(M,1);

n = 30; %resolution of parameter
a = [0.95,1,2,10];
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
end    
figure;

for p = 1:1:num_a
%set(gcf, 'Position',  [100, 100, 500, 400])
subplot(2,2,p)
%subplot(1,2,p)
surf(t1,t2,l(:,:,p))
hold on

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

%view([137,14])
view([57 24])
%view([134,13])
%view([131,19])
end