x=trainx(8,:);
X=[];
for i=1:28
    
   X=[X ;x(1+28*(i-1):28*i)];
end
 I=mat2gray(X');


y=noise_1_3(trainx(10,:)');
y=y';
Y=[];
for i=1:28
    
   Y=[Y ;y(1+28*(i-1):28*i)];
end
  J=mat2gray(Y');

figure
%  imshow(I)
 
  imshow(J)