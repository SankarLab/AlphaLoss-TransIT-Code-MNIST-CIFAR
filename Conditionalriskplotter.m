clc;
clear;

a = [0.5,0.65,2];
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

for j = 1:1:n
if j == 3
    plot(x,go,'LineWidth',2.5)
    hold on
    plot(x,fo(:,j),'LineWidth',2.5)
    hold on
else
    plot(x,fo(:,j),'LineWidth',2.5)
    hold on
end
end
plot(x,ho,'LineWidth',2.5)
rad = 15;

title('C_{\alpha}(\eta,f^{*}) of Margin-based \alpha-loss','FontSize',rad)
lgd = legend({'\alpha=.5 [exponential loss]','\alpha = .65','\alpha = 1 [log-loss]','\alpha = 2','\alpha = \infty [sigmoid loss]'},'FontSize',14,'Location','south');
lgd.FontSize = rad;
xlabel('\eta','FontSize',rad)
ylabel('C_{\alpha}(\eta,f^{*})','FontSize',rad)
