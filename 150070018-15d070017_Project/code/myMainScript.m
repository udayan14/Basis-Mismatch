%% CS 754 Advanced Image Processing
% Final Project
%
% This project has been done by:
% 
% * Udayan Joshi    150070018
% * Pranav Kulkarni 15D070017
% 

%% Effect of basis mismatch in compressive sensing

N = 100;
P = 1000;
Dict = zeros(P,N);
arr = linspace(0,1,P);
for i = 1:N
   Dict(:,i) = sin(2*pi*i*arr); 
end

M = 20;
phi_i = (2*(rand(M,P)<0.5) - 1)/sqrt(M);
A = phi_i * Dict;
%% Creating an off the grid signal

signal1 = sin(2*pi*11*arr) + sin(2*pi*20*arr) + 2*sin(2*pi*21*arr);
signal2 = sin(2*pi*11.3*arr) + sin(2*pi*20.2*arr) + 2*sin(2*pi*21.1*arr);
cs1 = phi_i * signal1';
cs2 = phi_i * signal2';

%Reconstruction using pseudo-inverse(no compressive measurements taken)

coeff1 = Dict \ signal1';
out1 = find(abs(coeff1) > 1e-4);
coeff2 = Dict \ signal2';
out2 = find(abs(coeff2) > 1e-4);

recon1 = Dict * coeff1;
err1 = norm(recon1 - signal1')/norm(signal1);
sprintf('The relative error for recontruction using pseudo inverse is %f',err1)
recon2 = Dict * coeff2;
err2 = norm(recon2 - signal2')/norm(signal2);
sprintf('The relative error for recontruction using pseudo inverse is %f',err2)

figure('Name','NO basis mismatch : Recovery using Pseudoinverse')
plot(arr,signal1);
hold on;
plot(arr,recon1);
title('Reconstruction results : No basis mismatch')
legend('Original Signal','Reconstruction using CS')

figure('Name','Basis mismatch : Recovery using Pseudoinverse')
plot(arr,signal2);
hold on;
plot(arr,recon2);
title('Reconstruction results : basis mismatch')
legend('Original Signal','Reconstruction with pinv')
%% Reconstruction using l1-ls minimization

lambda = 0.01;
rel_tol = 0.001;
[coeff1,status1]=l1_ls(A,cs1,lambda,rel_tol);
[coeff2,status2]=l1_ls(A,cs2,lambda,rel_tol);

recon1 = Dict * coeff1;
err1 = norm(recon1 - signal1')/norm(signal1);
sprintf('The relative error for recontruction using l1-ls minimization is %f',err1)
recon2 = Dict * coeff2;
err2 = norm(recon2 - signal2')/norm(signal2);
sprintf('The relative error for recontruction using l1-ls minimization is %f',err2)

figure('Name','NO basis mismatch : Recovery using L1 minimization')
plot(arr,signal1);
hold on;
plot(arr,recon1);
title('Reconstruction results : No basis mismatch')
legend('Original Signal','Reconstruction (L1 minimization)')

figure('Name','basis mismatch : Recovery using L1 minimization')
plot(arr,signal2);
hold on;
plot(arr,recon2);
title('Reconstruction results : basis mismatch')
legend('Original Signal','Reconstruction (L1 minimization)')
% We thus observe that under basis mismatch condition, the reconstruction error is much higher. 

%% Implementing the paper

Q = 2;      %%Increasing Q reduces reconstruction error
N = 256;
M = 128;
arr = 0:N-1;
dict = zeros(N,Q*N);
 
for i = 1 : Q*N
    if i <= Q*N/2
        dict(:,i) = cos(2*pi*(i-1)*arr/(Q*N))*(2/N)^0.5;
    else
        dict(:,i) = -1 * sin(2*pi*(Q*N-i)*arr/(Q*N))*(2/N)^0.5;
    end
end

s1 = 3*cos(2*pi*3*arr/(Q*N) + 3.7) + 4*cos(2*pi*7*arr/(Q*N) + 2.1) + 5*cos(2*pi*10*arr/(Q*N) + 2.123);
s2 = 3*cos(2*pi*3.11*arr/(Q*N) + 3.7) + 4*cos(2*pi*7.23*arr/(Q*N) + 2.1) + 5*cos(2*pi*10.37*arr/(Q*N) + 2.123);

phi = (2*(rand(M,N)<0.5) - 1)/sqrt(M);
A = phi * dict;

cs1 = phi * s1';
cs2 = phi * s2';

lambda = 0.01;
rel_tol = 0.00001;

[coeff1,~]=l1_ls(A,cs1,lambda,rel_tol);
recon1 = dict * coeff1;
err1 = norm(recon1 - s1')/norm(s1);
sprintf('The relative error for recontruction using l1-ls minimization is %f',err1)
figure('Name','Signal without basis mismatch')
plot(arr,s1);
hold on;
plot(arr,recon1);
title('Reconstruction results : No basis mismatch')
legend('Original Signal','Reconstruction using L1')

[coeff2,status1]=l1_ls(A,cs2,lambda,rel_tol);
recon2 = dict * coeff2;
err2 = norm(recon2 - s2')/norm(s2);
sprintf('The relative error for recontruction using l1-ls minimization is %f',err2)
figure('Name','Signal with basis mismatch')
plot(arr,s2);
hold on;
plot(arr,recon2);
title('Reconstruction results : basis mismatch')
legend('Original Signal','Reconstruction using L1')

%% Testing ACS without noise
[coeff1,freq1] = myACS(cs1,phi,N,Q);
[coeff2,freq2] = myACS(cs2,phi,N,Q);

recon1 = dict * coeff1;
err1 = norm(recon1 - s1')/norm(s1);
sprintf('The relative error for recontruction using ACS is %f',err1)
figure('Name','Signal without basis mismatch')
title('Reconstruction results without basis mismatch')
plot(arr,s1);
hold on;
plot(arr,recon1);
title('Reconstruction results : No basis mismatch')
legend('Original Signal','Reconstruction using ACS')


recon2 = dict * coeff2;
err2 = norm(recon2 - s2')/norm(s2);
sprintf('The relative error for recontruction using ACS is %f',err2)
figure('Name','Signal with basis mismatch')
plot(arr,s2);
hold on;
plot(arr,recon2);
title('Reconstruction results : basis mismatch')
legend('Original Signal','Reconstruction using ACS')

%% Testing ACS with noise

SNR = 10;
sigma1 = norm(s1)/((N^0.5)*(10^(SNR/20)));
sigma2 = norm(s2)/((N^0.5)*(10^(SNR/20)));

noisy_cs1 = cs1 + sigma1*randn(size(cs1));
noisy_cs2 = cs2 + sigma2*randn(size(cs2));

[coeff1,freq1] = myACS(noisy_cs1,phi,N,Q);
[coeff2,freq2] = myACS(noisy_cs2,phi,N,Q);

recon1 = dict * coeff1;
err1 = norm(recon1 - s1')/norm(s1);
sprintf('The relative error for recontruction using ACS is %f',err1)
figure('Name','Signal without basis mismatch')
plot(arr,s1);
hold on;
plot(arr,recon1);
title('Reconstruction results without basis mismatch')
legend('Original Signal','Reconstruction using ACS')

recon2 = dict * coeff2;
err2 = norm(recon2 - s2')/norm(s2);
sprintf('The relative error for recontruction using ACS is %f',err2)
figure('Name','Signal with basis mismatch')
plot(arr,s2);
hold on;
plot(arr,recon2);
title('Reconstruction results with basis mismatch')
legend('Original Signal','Reconstruction using ACS')

%% Plotting Graphs

    SNR_arr = [0 5 10 15 20 25 30 35 40];
    s1_err_arr = zeros(size(SNR_arr));
    s2_err_arr = zeros(size(SNR_arr));
    s1_spar_arr = zeros(size(SNR_arr));
    s2_spar_arr = zeros(size(SNR_arr));

    k = 0;

for SNR = SNR_arr
    k = k+1;
    sigma1 = norm(s1)/((N^0.5)*(10^(SNR/20)));
    sigma2 = norm(s2)/((N^0.5)*(10^(SNR/20)));

    new_cs1 = cs1 + sigma1*randn(size(cs1));
    new_cs2 = cs2 + sigma2*randn(size(cs2));
    
    [coeff1,freq1] = myACS(new_cs1,phi,N,Q);
    [coeff2,freq2] = myACS(new_cs2,phi,N,Q);
    recon1 = dict * coeff1;
    err1 = norm(recon1 - s1')/norm(s1);
    s1_err_arr(k) = err1;
    s1_spar_arr(k) = length(find(coeff1>0.1));
    % sprintf('The relative error for recontruction using ACS is %f',err1)
    % figure('Name','Signal without basis mismatch')
    % title('Reconstruction results without basis mismatch')
    % plot(arr,s1);
    % hold on;
    % plot(arr,recon1);
    % legend('Original Signal','Reconstruction using ACS')

    recon2 = dict * coeff2;
    err2 = norm(recon2 - s2')/norm(s2);
    s2_err_arr(k) = err2;
    s2_spar_arr(k) = length(find(coeff2>0.1));
    % sprintf('The relative error for recontruction using ACS is %f',err2)
    % figure('Name','Signal with basis mismatch')
    % title('Reconstruction results with basis mismatch')
    % plot(arr,s2);
    % hold on;
    % plot(arr,recon2);
    % legend('Original Signal','Reconstruction using ACS')
end

figure('Name','Reconstruction error vs SNR')
plot(SNR_arr,s1_err_arr);
hold on;
plot(SNR_arr,s2_err_arr);
title('Reconstruction error vs SNR')
legend('Signal without basis mismatch','Signal with basis mismatch')

figure('Name','Sparsity Level vs SNR')
plot(SNR_arr,s1_spar_arr);
hold on;
plot(SNR_arr,s2_spar_arr);
title('Sparsity Level  vs SNR')
legend('Signal without basis mismatch','Signal with basis mismatch')

%% Plots for M

Q = 1;      %%Increasing Q reduces reconstruction error
N = 256;
arr = 0:N-1;
dict = zeros(N,Q*N);

 
for i = 1 : Q*N
    if i <= Q*N/2
        dict(:,i) = cos(2*pi*(i-1)*arr/(Q*N))*(2/N)^0.5;
    else
        dict(:,i) = -1 * sin(2*pi*(Q*N-i)*arr/(Q*N))*(2/N)^0.5;
    end
end

s1 = 3*cos(2*pi*3*arr/(Q*N) + 3.7) + 4*cos(2*pi*7*arr/(Q*N) + 2.1) + 5*cos(2*pi*10*arr/(Q*N) + 2.123);
s2 = 3*cos(2*pi*3.666*arr/(Q*N) + 3.7) + 4*cos(2*pi*7.23*arr/(Q*N) + 2.1) + 5*cos(2*pi*10.37*arr/(Q*N) + 2.123);

M_arr = [120, 100, 80, 60, 40, 20];

s1_l1_err_arr = zeros(size(M_arr));
s2_l1_err_arr = zeros(size(M_arr));
s1_l1_spar_arr = zeros(size(M_arr));
s2_l1_spar_arr = zeros(size(M_arr));

s1_ACS_err_arr = zeros(size(M_arr));
s2_ACS_err_arr = zeros(size(M_arr));
s1_ACS_spar_arr = zeros(size(M_arr));
s2_ACS_spar_arr = zeros(size(M_arr));

k=0;
lambda = 0.01;
rel_tol = 0.00001;

for M = M_arr;
k= k+1;
phi = (2*(rand(M,N)<0.5) - 1)/sqrt(M);
A = phi * dict;

cs1 = phi * s1';
cs2 = phi * s2';

[coeff1,~]=l1_ls(A,cs1,lambda,rel_tol);
recon1 = dict * coeff1;
err1 = norm(recon1 - s1')/norm(s1);
s1_l1_err_arr(k) = err1;
s1_l1_spar_arr(k) = length(find(coeff1>0.1));

[coeff2,status1]=l1_ls(A,cs2,lambda,rel_tol);
recon2 = dict * coeff2;
err2 = norm(recon2 - s2')/norm(s2);
s2_l1_err_arr(k) = err2;
s2_l1_spar_arr(k) = length(find(coeff2>0.1));


[coeff1,freq1] = myACS(cs1,phi,N,Q);
[coeff2,freq2] = myACS(cs2,phi,N,Q);

recon1 = dict * coeff1;
err1 = norm(recon1 - s1')/norm(s1);
s1_ACS_err_arr(k) = err1;
s1_ACS_spar_arr(k) = length(find(coeff1>0.1));

recon2 = dict * coeff2;
err2 = norm(recon2 - s2')/norm(s2);

s2_ACS_err_arr(k) = err2;
s2_ACS_spar_arr(k) = length(find(coeff2>0.1));

end


figure('Name','Reconstruction error vs M for l1_ls')
plot(M_arr,s1_l1_err_arr);
hold on;
plot(M_arr,s2_l1_err_arr);
title('Reconstruction error vs M')
legend('Signal without basis mismatch','Signal with basis mismatch')

figure('Name','Sparsity Level vs M for l1_ls')
plot(M_arr,s1_l1_spar_arr);
hold on;
plot(M_arr,s2_l1_spar_arr);
title('Sparsity Level  vs M')
legend('Signal without basis mismatch','Signal with basis mismatch')



figure('Name','Reconstruction error vs M for ACS')
plot(M_arr,s1_ACS_err_arr);
hold on;
plot(M_arr,s2_ACS_err_arr);
title('Reconstruction error vs M')
legend('Signal without basis mismatch','Signal with basis mismatch')

figure('Name','Sparsity Level vs M for ACS')
plot(M_arr,s1_ACS_spar_arr);
hold on;
plot(M_arr,s2_ACS_spar_arr);
title('Sparsity Level  vs M')
legend('Signal without basis mismatch','Signal with basis mismatch')