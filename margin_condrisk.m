a = [.5,.75,1.1,5,200];
l = max(length(a));
M = 10000;
z = linspace(-5,5,M);
f = zeros(l,M);
g = zeros(1,M);
h = zeros(1,M);

for i = 1:1:M
    g(i) = log(1+exp(-z(i)));
end

for i = 1:1:M
    h(i) = 1/(1+exp(z(i)));
end    
    
for i = 1:1:l
    for j = 1:1:M
        %f(i,j) = ((a(i)/(a(i)-1))*(1 - (1./(1+exp(-z(j)/a(i)))).^(1-1/a(i))));
        f(i,j) = ((a(i)/(a(i)-1))*(1 - (1./(1+exp(-z(j)))).^(1-1/a(i))));
        %f(i,j) = (1+exp(z(j)))^(1/a(i))/((1+exp(z(j)))^(1/a(i)) + (1+exp(-z(j)))^(1/a(i)));
        %f(i,j) = log(exp((1-z(j))*a(i)) + 1)/a(i);
        %f(i,j) = (1+(exp(2*a(i)) + exp(a(i)*(1-z(j))))/(exp(2*a(i)) + exp(a(i)*(1+z(j)))))^(-1);
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
plot(z,f(5,:),'LineWidth',4)
hold on
%plot(z,f(6,:),'LineWidth',4)
%hold on
rad = 14;
rad1 = 17;
plot(z,h,'LineWidth',4)
hold on
title('(a) Margin-based \alpha-loss','FontSize',rad)
xlabel('Margin','FontSize',rad)
ylabel('Loss','FontSize',rad)
lgd = legend('\alpha = .3', '\alpha = .5 [Exponential]','\alpha = .77','\alpha = 1 [Logistic]','\alpha = 1.44','\alpha = +\infty [Sigmoid]','Location','northeast');
lgd.FontSize = rad1;
ylim([0, 5])
set(gca,'fontsize',20)

%%%%%%%%%%%%%%%%%

%a = [0.4,0.5,0.65,2];
l = max(length(a));
M = 10000;

x = linspace(0,1,M);

n = max(size(a));
fo = zeros(M,n);
go = zeros(M,1);
ho = zeros(M,1);

for j = 1:1:n
    for i = 1:1:M
        fo(i,j) =(a(j)/(a(j)-1))*(1 - ((x(i).^(a(j)/(a(j)-1) + a(j)))/(x(i).^(a(j)) + (1-x(i)).^(a(j)))).^(1-(1/a(j))) - (((1-x(i)).^(a(j)/(a(j)-1) + a(j)))/(x(i).^(a(j)) + (1-x(i)).^(a(j)))).^(1-(1/a(j))));
    end
end

for i = 1:1:M
    go(i) = -x(i)*log(x(i)) - (1-x(i))*log(1-x(i));
end

for i = 1:1:M
    if x(i) >= 1-x(i)
        ho(i) = 1 - x(i);
    else
        ho(i) = x(i);
    end
end
subplot(1,2,2);
for j = 1:1:n
if j == 4
    plot(x,go,'LineWidth',4)
    hold on
    plot(x,fo(:,j),'LineWidth',4)
    hold on
else
    plot(x,fo(:,j),'LineWidth',4)
    hold on
end
end
plot(x,ho,'LineWidth',4)
rad = 14;
rad1 = 16;

title('(b) Minimum Conditional Risk','FontSize',rad)
%lgd = legend({'\alpha = .4','\alpha=.5 [Exponential Loss]','\alpha = .65','\alpha = 1 [Logistic Loss]','\alpha = 2','\alpha = \infty [Sigmoid Loss]'},'FontSize',10,'Location','south');
%lgd.FontSize = rad1;
xlabel('\eta','FontSize',rad)
ylabel('C_{\alpha}(\eta,f^{*}) [Log-Scale]','FontSize',rad)
ylim([0, 1.75])
set(gca,'fontsize',20)
