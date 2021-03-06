function createfigure(X1, YMatrix1, Y1, l1, l2, l3)
%CREATEFIGURE(X1, YMatrix1, Y1)
%  X1:  vector of x data
%  YMATRIX1:  matrix of y data
%  Y1:  vector of y data

%  Auto-generated by MATLAB on 29-Oct-2021 15:42:14

% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% Create multiple lines using matrix input to plot
plot1 = plot(X1,YMatrix1,'Parent',axes1,'LineWidth',1);
set(plot1(1),'LineStyle','-.','Color',[1 0 0]);
set(plot1(2),'Color',[0 1 0]);
set(plot1(3),'LineStyle','--',...
    'Color',[0.87058824300766 0.490196079015732 0]);

% Uncomment the following line to preserve the X-limits of the axes
% xlim(axes1,[-1.1 1.1]);
% Uncomment the following line to preserve the Y-limits of the axes
ylim(axes1,[0 2.2]);
box(axes1,'on');
hold(axes1,'off');
% Set the remaining axes properties
set(axes1,'XColor',[0 0 0],'YColor',[0 0 0],'YTick',[0 0.5 1 1.5 2 2.5]);
% Create axes
axes2 = axes('Parent',figure1);
hold(axes2,'on');
colororder([0.494 0.184 0.556;0.466 0.674 0.188;0.301 0.745 0.933;0.635 0.078 0.184;0 0.447 0.741;0.85 0.325 0.098;0.929 0.694 0.125]);

% Create plot
plot(X1,Y1,'Parent',axes2,'LineWidth',1,'LineStyle',':','Color',[0 0 1]);

% Uncomment the following line to preserve the X-limits of the axes
% xlim(axes2,[-1.1 1.1]);
% Uncomment the following line to preserve the Y-limits of the axes
ylim(axes2,[0 1.6]);
hold(axes2,'off');
% Set the remaining axes properties
set(axes2,'Color','none','HitTest','off','XColor',[0 0 0],'YAxisLocation',...
    'right','YColor',[0 0 0],'YTick',[0 0.5 1 1.5]);
% Create textbox
annotation(figure1,'textbox',...
    [0.255427607968038,0.605539475745798,0.304947448327989,0.235148519909872],...
    'Color',[0.8 0 0],...
    'String',{sprintf('peak loss=%.4f',l1)},...
    'EdgeColor','none');

% Create textbox
annotation(figure1,'textbox',...
    [0.631790371410027,0.083530640355914,0.178879315581032,0.235148519909871],...
    'Color',[0 0 1],...
    'String',{'keypoint'},...
    'EdgeColor','none');

% Create textbox
annotation(figure1,'textbox',...
    [0.59663112557549,0.640686239621974,0.318247136419826,0.22093023731067],...
    'Color',[0 0.498039215803146 0],...
    'String',{sprintf('peak loss=%.4f',l2)},...
    'EdgeColor','none');

% Create textbox
annotation(figure1,'textbox',...
    [0.595423071596731,0.415858983920567,0.318247136419826,0.235148519909871],...
    'Color',[0.87058824300766 0.490196079015732 0],...
    'String',{sprintf('peak loss=%.4f',l3)},...
    'FitBoxToText','off',...
    'EdgeColor','none');

