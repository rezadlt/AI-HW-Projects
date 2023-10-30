clc
clear
n=100;
min=-10;
max=10;
A=0; % noise
x=min:(max-min)/(n-1):10;
x1=x+A*rand(size(x));
% x=x1;
y1=2*x;
y2=sin(x);
y3=2*x.^2+4*x+5;
y4=tan(x)+x;
y5=3*x.^4-2*x.^3-10;
y6=5*sin(x).^2+0.2*x.^2-6;






% plot(x,y1)