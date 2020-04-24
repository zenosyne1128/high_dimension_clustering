%% 4
load 'digits.mat'
 r = 0; affine = false; outlier = true; rho = 1; alpha = 20;
tic
[missrate,y_pred,C] = SSC(data',r,affine,alpha,outlier,rho,target'+1);
toc %事实上我们在SSC函数内部调用了tic,toc，最终使用的是内部调用时间
ssc = y_pred';
%% 3
load 'wine.mat'
r = 0; affine = true; outlier = true; rho = 8; alpha = 3800;
tic
[missrate,y_pred,C] = SSC(data',r,affine,alpha,outlier,rho,target'+1);
ssc = y_pred';
%% 2   作废
load '1.mat'         %载入2015研究生数模B.1数据集
a=ones(40,1);
b=ones(100,1) .* 2;
c=ones(60,1);
y_true = [a;b;c];    %制作标签
X = data';
y = y_true';
save indsp1.mat X y
  %给pycharm准备数据

r = 0; affine = false; outlier = false; rho = 1; alpha = 200;
tic
[missrate,y_pred,C] = SSC(data,r,affine,alpha,outlier,rho,y_true);
ssc = y_pred';
save ssc
%% 2
load 'unsp.mat'
 r = 0; affine = false; outlier = true; rho = 1; alpha = 20;
tic
[missrate,y_pred,C] = SSC(data',r,affine,alpha,outlier,rho,target'+1);
toc %事实上我们在SSC函数内部调用了tic,toc，最终使用的是内部调用时间
ssc = y_pred';
%% 1
load 'synthetic.mat'
r = 0; affine = false; outlier = true; rho = 1; alpha = 13;
tic
[missrate,y_pred,C] = SSC(data',r,affine,alpha,outlier,rho,target'+1);
toc %事实上我们在SSC函数内部调用了tic,toc，最终使用的是内部调用时间
ssc = y_pred';