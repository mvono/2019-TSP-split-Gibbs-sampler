
%-------------------------------------------------------------------------%
%                    SPLIT-AND-AUGMENTED GIBBS SAMPLER (SPA)              %
%                         APPLIED TO IMAGE DECONVOLUTION                  %
%                            ON THE LENA IMAGE                            %
%-------------------------------------------------------------------------%
% File: SPA_lena.m
% Author: M. VONO
% Created on: 16/05/2018
% Last modified : 16/05/2018
clearvars;
close all;
addpath('../utils/'); % to use HXconv function and to load lena
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
    [X_MC,Z_MC,U_MC] = SPA(D,mu1,FB,F2B,rho,alpha,y,FBC,gamma,F2L,N,N_MC);
%-------------------------------------------------------------------------%
%%
%-------------------------------------------------------------------------%
% Display PSNR and SNR associated to the MMSE estimator of x
[PSNR, SNR] = ...
            psnr(uint8(mean(X_MC(:,:,N_bi:N_MC),3)), uint8(refl));
disp(['PSNR: ' num2str(PSNR) ' dB']);
disp(['SNR: ' num2str(SNR) ' dB']);
%-------------------------------------------------------------------------%
%%
%-------------------------------------------------------------------------%
% Plot the results                                                        
plot_RESULT(y,refl,X_MC,Z_MC,U_MC,N_bi,N_MC,N);   
%-------------------------------------------------------------------------%