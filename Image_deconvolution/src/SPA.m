
%-------------------------------------------------------------------------%
%                  SPLIT-AND-AUGMENTED GIBBS SAMPLER                      %
%-------------------------------------------------------------------------%

function [X_MC,Z_MC,U_MC] = SPA(D,mu1,FB,F2B,rho,alpha,y,FBC,gamma,F2L,...
                                N,N_MC)

%-------------------------------------------------------------------------%
% This function computes the SPA algorithm to solve the linear inverse 
% problem y = H*x + n associated to the image deconvolution problem.

    % INPUTS:
        % D: precision matrix associated to the likelihood.
        % mu1: hyperparameter used in the AuxV1 algorithm (see Marnissi 
        % et al., 2017).
        % FB: counterpart of the blur operator in the Fourier domain.
        % F2B: same as FB with coefficients equal to |FB|.^2.
        % rho: user-defined standard deviation of the variable of 
        %      interest x.
        % alpha: user-defined hyperparameter of the prior p(u).
        % y: observations (2D-array).
        % FBC: conjugate of FB.
        % gamma: regularization parameter.
        % F2L: same as FL with coefficients equal to |FL|.^2 (with FL the 
        % counterpart of the matrix L used as regularization in the Fourier
        % domain).
        % N: dimension of x (2-D array).
        % N_MC : total number of MCMC iterations.
        
    % OUTPUT:
        % X_MC,Z_MC,U_MC: samples (2D-array) from the joint 
        % posterior.
%-------------------------------------------------------------------------%

tic;
disp(' ');
disp('BEGINNING OF THE SAMPLING');

%-------------------------------------------------------------------------
% Initialization
    % define matrices to store the iterates
    X_MC = zeros(N,N,N_MC);
    Z_MC = zeros(N,N,N_MC);
    U_MC = zeros(N,N,N_MC);
    % initialize the latter matrices
    X_MC(:,:,1) = rand(N,N)*255;
    Z_MC(:,:,1) = rand(N,N)*255;
    U_MC(:,:,1) = rand(N,N)*255;
%-------------------------------------------------------------------------

%-------------------------------------------------------------------------
% Gibbs sampling
h = waitbar(0,'Sampling in progress...');
for t = 1:(N_MC-1)
    
     % 1. Sampling x with the method AuxV1
        
        % 1.1. Sampling the auxiliary variable v1
        moy = (1 / mu1 - D) .* real(ifft2(FB .* fft2(X_MC(:,:,t)))); % real
        moy = reshape(moy,[1,N^2]);
        sigma = 1 / mu1 - reshape(D,[1,N^2]);
        v1 = mvnrnd(moy,sigma)';
        v1 = reshape(v1,[N,N]);
        
        % 1.2. Sampling the variable of interest x
        z0 = fft2(Z_MC(:,:,t));
        u0 = fft2(U_MC(:,:,t));
        precision = (1 / mu1) * F2B  + (1 / rho^2);
        moy = (FBC .* fft2(D .* y) ...
               + (1 / rho^2) * (z0 - u0) ...
               + FBC .* fft2(v1)) ./ precision;
        eps = sqrt(0.5) * (randn(N,N) + sqrt(-1)*randn(N,N));
        x0 = moy + eps ./ sqrt(precision);
        X_MC(:,:,t+1) = real(ifft2(x0));

        % 2.2.2. Sampling z
        precision = gamma * reshape(F2L,[N^2,1]) + (1 / rho^2);
        x0 = reshape(fft2(X_MC(:,:,t+1)),[N^2,1]);
        u0 = reshape(fft2(U_MC(:,:,t)),[N^2,1]);
        moy = (1 / rho^2) * (x0 + u0) ./ precision;
        eps = sqrt(0.5) * (randn(N^2,1) + sqrt(-1)*randn(N^2,1));
        z0 = moy + eps ./ sqrt(precision);
        Z_MC(:,:,t+1) = real(ifft2(reshape(z0,[N,N])));
        clear eps x0 u0 z0;

        % 2.2.3. Sampling u        
        cov = (alpha^2 * rho^2) / (alpha^2 + rho^2);
        moy = fft2(Z_MC(:,:,t+1)-X_MC(:,:,t+1)) * alpha^2 / (rho^2 + alpha^2);
        moy = reshape(moy,[N^2,1]);
        eps = sqrt(0.5) * (randn(N^2,1) + sqrt(-1)*randn(N^2,1));
        u0 = moy + eps .* sqrt(cov);
        U_MC(:,:,t+1) = real(ifft2(reshape(u0,[N,N])));
        clear moy cov eps u0;
    
    % Show iteration counter
    waitbar(t/N_MC);
    
end
%-------------------------------------------------------------------------

t_1 = toc;
close(h);
disp('END OF THE GIBBS SAMPLING');
disp(['Execution time of the Gibbs sampling: ' num2str(t_1) ' sec']);

end