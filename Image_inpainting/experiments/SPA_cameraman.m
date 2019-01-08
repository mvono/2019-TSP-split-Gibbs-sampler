
%-------------------------------------------------------------------------%
%                    SPLIT-AND-AUGMENTED GIBBS SAMPLER (SPA)              %
%                         APPLIED TO IMAGE INPAINTING                     %
%                            ON THE CAMERAMAN IMAGE                       %
%-------------------------------------------------------------------------%
% File: SPA_cameraman.m
% Author: M. VONO
% Created on: 16/05/2018
% Last modified : 16/05/2018
clearvars;
close all;
addpath('../utils/'); % to use E-PO and P-MYULA within SPA
addpath('../src/'); % to launch SPA
%-------------------------------------------------------------------------%
% REF.                                                                    %
% M. VONO et al.,                                                         %
% "Split-and-augmented Gibbs sampler - Application to large-scale         %
% inference problems", submitted, 2018.                                   %
%-------------------------------------------------------------------------%

%-------------------------------------------------------------------------%
% Load workspace variables (go to ../utils/initial_param_SPA.m to 
% modify them) and launch SPA algorithm.                                 
    load('../utils/initial_param_SPA.mat'); 
    [X_MC,Z_MC,U_MC] = SPA(y_signal,Hmat,sigma,rho,beta,alpha,...
                                                        N,M,invQ,N_MC);
%-------------------------------------------------------------------------%
%%
%-------------------------------------------------------------------------%
% Display SNR, MSE and SSIM associated to the MMSE estimator of x
[isnr,mse] = ISNR(X,Y,X_MC,N_bi);
SSIM = ssim(mean(X_MC(:,:,N_bi:end),3),X);
disp(['ISNR: ' num2str(isnr) ' dB']);
disp(['MSE: ' num2str(mse)]);
disp(['SSIM: ' num2str(SSIM)]);
%-------------------------------------------------------------------------%
%%
%-------------------------------------------------------------------------%
% Plot the results                                                        
plot_RESULT(Y,X,X_MC,Z_MC,U_MC,N_bi,N_MC,N);     
%-------------------------------------------------------------------------%