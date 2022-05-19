a = [.3,.5,.77,1.44];
n = max(length(a));
M = 10000;
z = linspace(-5,5,M);
f = zeros(n,M);
g = zeros(1,M);
h = zeros(1,M);
l = zeros(M,n);

for i = 1:1:M
    g(i) = log(1+exp(-z(i)));
end

for i = 1:1:M
    h(i) = 1/(1+exp(z(i)));
end    
    
for i = 1:1:n
    for j = 1:1:M
        f(i,j) = (a(i)/(a(i)-1))*(1 - (1./(1+exp(-z(j)))).^(1-1/a(i)));
    end
end
figure1=figure('Position', [100, 100, 1200, 350]);

subplot(1,2,1);
plot(z,f(1,:),'LineWidth',4)
hold on
plot(z,f(2,:),'LineWidth',4)
hold on
plot(z,f(3,:),'LineWidth',4)
hold on
plot(z,g,'LineWidth',4)
hold on
plot(z,f(4,:),'LineWidth',4)
hold on
rad = 14;
rad1 = 17;
plot(z,h,'LineWidth',4)
title('(a) Margin-based \alpha-loss','FontSize',rad)
xlabel('Margin','FontSize',rad)
ylabel('Loss','FontSize',rad)
lgd = legend('\alpha = .3', '\alpha = .5 [Exponential]','\alpha = .77','\alpha = 1 [Logistic]','\alpha = 1.44','\alpha = +\infty [Sigmoid]','Location','northeast');
lgd.FontSize = rad1;
ylim([0, 5])
set(gca,'fontsize',20)

%%%%%%%%%%%%%%%%%

%a = [0.4,0.5,0.65,2];
a = [.3,.5,.77,.999999,1.44,1000000000000];
n = max(length(a));
for j = 1:1:n
    for i = 1:1:M
        l(i,j) = - sigmoid(z(i))*sigmoid(-z(i))*sigmoid(z(i))^(-1/(a(j)));
        %f(i,j) = sigmoid(x(i))*sigmoid(-x(i))*sigmoid(x(i))^(-1/(a(j)));
    end
end
subplot(1,2,2);
for i = 1:1:n
    plot(z,l(:,i),'LineWidth',4)
    hold on
end
title('(b) Derivative of Margin \alpha-loss','FontSize',rad)
%lgd = legend({'\alpha = .4','\alpha=.5 [Exponential Loss]','\alpha = .65','\alpha = 1 [Logistic Loss]','\alpha = 2','\alpha = \infty [Sigmoid Loss]'},'FontSize',10,'Location','south');
%lgd.FontSize = rad1;
xlabel('Margin (z)','FontSize',rad)
%ylabel('-l(\alpha,z)','FontSize',rad)
ylim([-2, 0])
xlim([-5,5])
set(gca,'fontsize',20)

function z = sigmoid(x)
z = 1/(1+exp(-x));
end