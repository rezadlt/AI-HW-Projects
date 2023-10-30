clc
clear
[TrainX,TrainLabel]=LoadData('train-images.idx3-ubyte','train-labels.idx1-ubyte');
[TestX,TestLabel]=LoadData('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte');
% I=mat2gray(TestX(:,:,1));
% figure
% imshow(I)
% k=6;
% I=mat2gray(TrainX(:,:,k));
% figure
% imshow(I)
% TrainLabel(k)

n=10000;
TrainX=double(reshape(TrainX,28*28,size(TrainX,3))'); %Train Data [60000x784]
TestX=double(reshape(TestX,28*28,size(TestX,3))'); %Test Data [10000x784]
trainx=TrainX(1:n,:)/255;
trainy=TrainLabel(1:n,:);
yout=zeros(length(trainy),10);
for (i=1:length(trainy))
   if (trainy(i)>0) 
   yout(i, trainy(i))=1;
   end
  if (trainy(i)==0) 
   yout(i,10)=1;
   end 
end
y1=yout(:,1);
y2=yout(:,2);
y3=yout(:,3);
y4=yout(:,4);
y5=yout(:,5);
y6=yout(:,6);
y7=yout(:,7);
y8=yout(:,8);
y9=yout(:,9);
y0=yout(:,10);
