clc
clear
[TrainX,TrainLabel]=LoadData('train-images.idx3-ubyte','train-labels.idx1-ubyte');
[TestX,TestLabel]=LoadData('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte');
% I=mat2gray(TestX(:,:,1));
% figure
% imshow(I)
%  k=6;
%  I=mat2gray(TrainX(:,:,k));
%  J = imnoise(I, 'gaussian', 0.1, 0.02);
% subplot(2,1,1);title('org')
% imshow(I)
% 
% subplot(2,1,2);title('noisy')
% 
%  imshow(J)



% 
% figure
% imshow(I(5).org)
% figure
% imshow(I(5).noise)

% imshow(I)

% imshow(J)

n=1000;
TrainX=double(reshape(TrainX,28*28,size(TrainX,3))'); %Train Data [60000x784]
TestX=double(reshape(TestX,28*28,size(TestX,3))'); %Test Data [10000x784]
Trainx=TrainX(1:n,:);
for i=1:n
 trainx(i,:)= imnoise(Trainx(i,:), 'gaussian', 0.1, 0.05);
 
    
end

