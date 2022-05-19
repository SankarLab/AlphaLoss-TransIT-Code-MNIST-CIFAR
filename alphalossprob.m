clc;
clear;
y = linspace(0,20,21);
p = .5;
m = 20;
M = max(size(y));
a = [0.35,.5,.7,.99,1.8,10000];
k = max(size(a));
f = zeros(M,k);
b = zeros(k,1);


% for i = 1:1:k
%     for j = 1:1:M
%         f(j,i) = (nchoosek(20,y(j))*p^(y(j))*p^(20-y(j)))^(a(i));
%     end
%     b(i) = sum(f(:,i));
%     for j = 1:1:M
%         f(j,i) = f(j,i)/b(i);
%     end
% end
% 
% figure1=figure('Position', [100, 100, 1200, 350]);
% 
% subplot(1,2,2);
% %subplot(211); 
% for i = 1:1:k
%     if (i < 5)
%         plot(y,log10(f(:,i)),'-.o','LineWidth',3)
%         hold on
%     else
%         plot(10,log10(1),'-ok','LineWidth',3)
%    end
%     elseif (i == 3)
%         plot(y,log10(f(:,i)),'-ok','LineWidth',3)
%         hold on
%     else
%         plot(y,log10(f(:,i)),'-o','LineWidth',3)
%         hold on
%     end
%     if i == 5
%         plot(y,log10(f(:,i)),'-ok','LineWidth',3)
%         hold on
%     end
% end
rad = 50;
rad1 = 16;
% xlabel('Realization from the Binomial Distribution','FontSize',rad)
% ylabel('Probability [Log Scale]','FontSize',rad)
% lgd = legend({'\alpha = 0.01', '\alpha = .5', '\alpha = 1 [Log-Loss]', '\alpha = 3','\alpha = \infty [0-1 Loss]'},'FontSize',14,'Location','south');
% lgd.FontSize = rad1;
% title('(b) Illustration of \alpha-Tilted Distribution','FontSize',rad)
% ylim([-6.5, 0])
% set(gca,'fontsize',16)


M = 1000;
x = linspace(.001,1,M);
%a = [0.35,0.5,.7,.999999,1.8,100000];
%a = [.9999,1.6,5]; 
%a = .9999999
n = max(size(a));
f = zeros(M,n);
k = zeros(M,1);


for j = 1:1:n
    for i = 1:1:M
        %f(i,j) = a(j)/(a(j)-1)*(1-((x(i)^a(j))/(x(i)^a(j) + (1-x(i))^a(j)))^(1-1/a(j)));
        f(i,j) = (a(j)/(a(j)-1)*(1-x(i)^(1-1/(a(j)))));
    end
end
%subplot(1,2,1);
for i = 1:1:n
    plot(x,f(:,i),'LineWidth',4)
    hold on
end
% 
%rad = 1;

xlabel('Probability','FontSize',rad)
ylabel('Loss','FontSize',rad)
lgd = legend({'\alpha = 0.35','\alpha=0.5','\alpha = 0.7','\alpha = 1 [Log-Loss]', '\alpha = 1.8','\alpha = \infty [0-1 Loss]'},'FontSize',14,'Location','northeast');
lgd.FontSize = rad1;
title('\alpha-loss','FontSize',rad)
ylim([0, 5])
set(gca,'fontsize',16)
