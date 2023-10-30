% clc
% clear
% close
i=1;
x=trainx(i,:)';
g=im_y_5_15(x);
if (g<0.5)
    g=0;
else
    g=1;
end

