x=105:5:135;%x轴上的数据，第一个值代表数据开始，第二个值代表间隔，第三个值代表终止
Gloucester=[0.976,0.977,0.9765,0.9768,0.9763,0.9761,0.9759]
Shuguang=[0.97,0.971,0.973,0.974,0.976,0.9741,0.9739]
Sardinia=[0.962,0.964,0.9663,0.968,0.972,0.9715,0.969]
wuhan=[0.970777778,0.9706,0.9715,0.971322222,0.971877778,0.971322222,0.971266667]

plot(x,Gloucester,'-o','color',[0.8 0 0],'LineWidth',1.5)
hold on
plot(x,Shuguang,'-x','color',[0 0 0.8],'LineWidth',1.5)
hold on
plot(x,Sardinia,'-^','color',[0.6 0 0.6],'LineWidth',1.5)
hold on
plot(x,wuhan,'-d','color',[0 0.8 0.5],'LineWidth',1.5)
hold on

axis([105,135,0.92,1])  %确定x轴与y轴框图大小
set(gca, 'FontSize',8,'FontWeight','bold') % 设置坐标轴字体是 8

% set(gca,'xLim',[105:5:135]);
% set(gca,'YLim',[0.92:0.2:1]);
% set(gca,'XTick',[105:5:135]) %x轴范围1-6，间隔1
% set(gca,'YTick',[0.92:0.2:1]) %y轴范围0-700，间隔100

xlabel('The values of \phi','FontWeight','bold')  %x轴坐标描述
ylabel('OA','FontWeight','bold') %y轴坐标描述
legend('Gloucester','Shuguang','Sardinia','wuhan','southwest','FontSize',8,'FontWeight','bold');   %右上角标注


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% x=105:5:135;%x轴上的数据，第一个值代表数据开始，第二个值代表间隔，第三个值代表终止
% Gloucester=[0.799,0.815,0.83,0.842,0.871,0.867,0.859];
% Shuguang=[0.655,0.71,0.72,0.74,0.75,0.736,0.725];
% Sardinia=[0.75,0.755,0.76,0.763,0.77,0.742,0.728];
% wuhan=[0.67,0.6987,0.6912,0.6998,0.7074,0.6953,0.6968];
% 
% plot(x,Gloucester,'-o','color',[0.8 0 0],'LineWidth',1.5)
% hold on
% plot(x,Shuguang,'-x','color',[0 0 0.8],'LineWidth',1.5)
% hold on
% plot(x,Sardinia,'-^','color',[0.6 0 0.6],'LineWidth',1.5)
% hold on
% plot(x,wuhan,'-d','color',[0 0.8 0.5],'LineWidth',1.5)
% hold on
% 
% axis([105,135,0.5,1])  %确定x轴与y轴框图大小
% set(gca, 'FontSize',8,'FontWeight','bold') % 设置坐标轴字体是 8
% 
% % set(gca,'xLim',[105:5:135]);
% % set(gca,'YLim',[0.92:0.2:1]);
% % set(gca,'XTick',[105:5:135]) %x轴范围1-6，间隔1
% % set(gca,'YTick',[0.92:0.2:1]) %y轴范围0-700，间隔100
% 
% xlabel('The values of \phi','FontWeight','bold')  %x轴坐标描述
% ylabel('KC','FontWeight','bold') %y轴坐标描述
% legend('Gloucester','Shuguang','Sardinia','wuhan','southwest','FontSize',8,'FontWeight','bold');   %右上角标注
