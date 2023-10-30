clc
clear
n=500;
min1=-10;
max1=10;
min2=-10;
max2=10;
min3=0;
max3=10;
A=0; % noise
x1=min1:(max1-min1)/(n-1):10;
x2=min1:(max1-min1)/(n-1):10;
x3=min1:(max1-min1)/(n-1):10;
% x=x1;
y1=2*x1+x2-x3;
y2=sin(x1)+sin(x2);

y3=abs(2*x1.^2+4*x1+5-3*x2.^2+3*sqrt(x3));
y4=tan(x1)+3*cot(x2)-2*x3.^3;
y5=3*x1.^4-2*x2.^3-10+5*x3-2;
y6=abs(5*sin(x1).^2+0.2*x3.^2-6*sqrt(x2.^5));
x=[x1;x2;x3];
% plot3(x1,x2,y6)
% 
% xlabel('x');ylabel('y'),zlabel('z')




