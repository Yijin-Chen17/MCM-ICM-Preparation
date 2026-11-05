% MATLAB 官方整理
%https://ww2.mathworks.cn/products/matlab/plot-gallery.html?s_tid=srchtitle_gallery_1
% 绘图案例
%% 折线图
%% 图框的基本设置
clc;clear;close all;
load('color_list.mat')

load('折线图测试数据.mat')

figure('Position',[100,100,600,400])  %整个框大小设置

x1=[1:100];
y1=2*x1+10*rand(1,length(x1));
y2=0.01*x1.^2+0.5*x1+10*rand(1,length(x1));

color_index=[1,3,4,7,8,9];  %颜色选取

fu={'-*','-^','-o','-d','-p'};   %符号

% %单根折线图  
% plot(x1,y1,'--','LineWidth',2,'Color',color_list(color_index(1),:))
% hold on      %保持在一个图窗内
% plot(x1,y2,'--','LineWidth',2,'Color',color_list(color_index(2),:))
% hold on 

color_set=[0.1451    0.6118    0.1451;1.0000    0.4980    0.0549];
% 单根折线图  
plot(x1,y1,'--','LineWidth',2,'Color',color_set(1,:))
hold on      %保持在一个图窗内
plot(x1,y2,'-','LineWidth',1.5,'Color',color_set(2,:))
hold on 

xlabel('time')
ylabel('value')
title('title')

xticks([10:20:100]);
% xticks([10,30,50,70,90]);

xticklabels({'1-10','1-20','2-10','2-20','3-10'})

%图框全框或者半框的设置
box off
%box on

% x轴和y轴的范围设置

% xlim([0,100]) 
% ylim([0,100])

%直线的设置
xline(30,'--','LineWidth',1.2)
yline(30)

%网格的设置
% grid("on")
% set(gca,'XGrid','on',GridLineStyle',':','GridColor','k','GridAlpha',1);

% set(gca,'xgrid','on');
% set(gca,'ygrid','on');

legend('sybol1','sybol2')
legend('Box','off')

set(gca,'FontName','Times New Roman',"FontSize",12,"LineWidth",1.2)

% set(gca,'looseInset',[0 0 0 0]); %去除图片的白边


%%
% 导入颜色矩阵
clc;clear;close all;
load('color_list.mat')
load('折线图测试数据.mat')
%
color_index=[1,3,4,7,8,9];  %颜色选取
fu={'-*','-^','-o','-d','-p'};   %符号
%单根折线图  
figure
plot(data_test4(1,:),fu{1,5},'LineWidth',2,'Color',color_list(color_index(1),:))
hold on 
plot(data_test4(2,:),fu{1,4},'LineWidth',2,'Color',color_list(color_index(2),:))
set(gca,"FontSize",11,"LineWidth",1.2)
xlabel('时间')
ylabel('数值')
title('标题自定义')
box off
legend('对象一','对象二')

%%
%多根折线图

figure('Position',[100,200,650,350])
str=[];

color_index=[3,6,10,50,60];
color_set=color_list(color_index,:);

for i=1:5
   plot(data_test4(i,:),fu{1,i},'LineWidth',2,'Color',color_set(i,:))
   hold on  
   str{1,i}=['object',num2str(i)];
end
set(gca,'FontName','Times New Roman',"FontSize",12,"LineWidth",1.2)

% box off

xlabel('x')
ylabel('y')
title('title')
% str={'A','B','C','D','E'};
lgb=legend(str);

lgb.NumColumns=3;
legend('Box','on','EdgeColor',[0.8,0.8,0.8])
% legend('Box','on','Color',[0.95,0.95,0.95])


%%  单个误差带的折线图

data_test=[1,2,3,4,5;1.5,1.8,2.3,3.2,4.5;0.2,2.2,3.1,3.3,4.6];
dmean=mean(data_test);
dstd=std(data_test);

x=[0.1,2.1,3.1,4.1,5.1]; %x轴数据

y1=dmean;  %y轴数据

low1=dmean-dstd;  %数据上限
high1=dmean+dstd;  %数据下限


figure('Position',[200,200,600,350])



color_set=[1.0000    0.6863    0.3490];

plot(x,y1,'-s','Color',color_set(1,:),'LineWidth',1.5)
hold on 
h1=fill([x,fliplr(x)],[low1,fliplr(high1)],'r');

hold on

h1.FaceColor = color_set(1,:);%定义区间的填充颜色   

h1.EdgeColor =[1,1,1];%边界颜色设置为白色

alpha (0.3)   %设置透明色

scatter(x,data_test,20,'o','filled','MarkerFaceColor',[0.5,0.5,0.5],'MarkerEdgeColor',[0.5,0.5,0.5])
hold on

set(gca,'FontName','Times New Roman',"FontSize",12,"LineWidth",1.1)

% box off
legend('mean','std','data')

xlabel('x')
ylabel('y')
title('title')



%% 绘制饼图
load('color_cell1.mat')
mycolor1=color_cell1{1,1};
% data=[0.160784313725490   0.615686274509804   0.572549019607843  0.672549019607843  0.872549019607843];
data=[1,2,3,6,7];
figure
pie3(data,ones(1,length(data)))
colormap(mycolor1)
set(gca,'looseInset',[0 0 0 0]); %去除图片的白边

%%
figure
labels=["species1 48% ","species2 35% ","species3 17% "];
pie3([48,35,17],labels)
color1_set=[  0.0039    0.4471    0.7373
    0.4706    0.6706    0.1882
    0.8471    0.3255    0.0980];
set(gca,'looseInset',[0 0 0 0]); %去除图片的白边
colormap(color1_set)


%% 柱状图
clc;clear;close all;
% load('color_list.mat')
% color=color_list(1:6,:);
color=[0.717647058823529,0.682352941176471,0.741176470588235;...
    0.807843137254902,0.792156862745098,0.890196078431373;0.580392156862745,...
    0.525490196078431,0.729411764705882;0.392156862745098,0.345098039215686,...
    0.470588235294118;0.560784313725490,0.533333333333333,0.741176470588235;...
    0.372549019607843,0.282352941176471,0.600000000000000;0.0156862745098039,...
    0.0196078431372549,0.0156862745098039];% 颜色数据

% color=[0.741176470588235,0.729411764705882,0.725490196078431;0.525490196078431,...
%     0.623529411764706,0.752941176470588;0.631372549019608,0.803921568627451,...
%     0.835294117647059;0.588235294117647,0.576470588235294,0.576470588235294;...
%     0.0745098039215686,0.407843137254902,0.607843137254902;0.454901960784314,...
%     0.737254901960784,0.776470588235294;0.0156862745098039,0.0196078431372549,0.0156862745098039];% 颜色

y=[28,34,18,13];%柱状图纵坐标
x=[1,2,3,4];     %柱状图横坐标

%%初始化显示位置 [x,y,dx,dy] 从 x,y坐标开始，dx,dy为沿升的长度
figure('Units', 'pixels', ...
    'Position', [100 100 460 275]);

width=0.7; %柱状图宽度 0~1
for  i=1:length(y)
   %bar(i,y(i),width) width 柱状图宽占比  ，FaceColor 柱状图填充颜色 ，EdgeColor 柱状图边缘颜色
   % LineWidth  边缘线宽 
   set(bar(i,y(i),width),'FaceColor',color(i,:),'EdgeColor',color(i+3,:),'LineWidth',2)
   hold on
end

%绘制虚线
ylabel('Time on Warm Floor(%)')

xticks([1 2 3 4]);
xticklabels({'objedct1','objedct2','objedct3','objedct4'})
box off
%坐标轴字体和大小
set(gca,"FontName","Times New Roman","FontSize",12,"LineWidth",2)
title("TPT Timecourse","FontName","Times New Roman","FontSize",12,"LineWidth",2);


%%  热图
clc;
clear;
close all;
%X为0-1矩阵 
X = rand(13);
%获取矩阵的列数
X = round(X,2);
%制造全是1的向量
B = ones(1,size(X,1));
% 替换X矩阵对角元素，使其均为1
X(logical(eye(size(X))))=B;
data=X;
load('color_cell1.mat')
mycolor1=color_cell1{1,1};
% 开始绘制热图
% 命名所有变量名字，这里大家可以替换成自己需要的变量名
label_name = {'N1','N2','N3','N4','N5','N6','N7','N8','N9','N10','N11','N12','N13'};
xlabel_name = label_name;
ylabel_name = label_name;
% 热图函数为heatmap;开始绘制
hot_figure = heatmap(xlabel_name,ylabel_name,data,'FontName','Times New Roman','FontSize',10);
hot_figure.GridVisible = 'off';
% 设置坐标区名字与图的大标题
hot_figure.Title = 'Heated';
hot_figure.XLabel = 'xlabel';
hot_figure.YLabel = 'ylabel';
colormap(gca,mycolor1);

