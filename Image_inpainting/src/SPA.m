
%-------------------------------------------------------------------------%
%                  SPLIT-AND-AUGMENTED GIBBS SAMPLER                      %
%-------------------------------------------------------------------------%

function [X_MC,Z_MC,U_MC] = SPA(y_signal,Hmat,sigma,rho,beta,alpha,...
                                                        N,M,invQ,N_MC)

%-------------------------------------------------------------------------%
% This function computes the SPA algorithm to solve the linear inverse 
% problem y = H*x + n associated to the image inpainting problem.

    % INPUTS:
        % y_signal: noisy observation (1D array).
        % Hmat: direct operator in the linear inverse problem y = H*x + n.
        % sigma: user-defined standard deviation of the noise.
        % rho: user-defined standard deviation of the variable of 
        %      interest x.
        % beta: user-defined hyperparameter of the prior p(z).
        % alpha: user-defined hyperparameter of the prior p(z).
        % N,M: respectively, the dimension of X (2D-array) and y
        % (1D-array).
        % invQ: pre-computed covariance matrix involved in the posterior
        % distribution of the variable of interest x.
        % N_MC: total number of MCMC iterates.
        
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
    
    % 1. Sample x from p(x|z,u,y) using Exact Perturbation-Optimization 
    % (E-PO) method.
    X_MC(:,:,t+1) = EPO(y_signal,Hmat,sigma,U_MC(:,:,t),Z_MC(:,:,t),...
                             rho,N,M,invQ);
    
    % 2. Sample z from p(z|x,u) using P-MYULA 
    % (see Durmus et al., 2018).
    [Z_MC(:,:,t+1)] = PMYULA(Z_MC(:,:,t),X_MC(:,:,t+1),...
                             U_MC(:,:,t),rho,beta,N);

    % 3. Sample d from p(u|x,z)
    x = reshape(X_MC(:,:,t+1),[N^2,1]);
    z = reshape(Z_MC(:,:,t+1),[N^2,1]);
    moy = (alpha^2 / (rho^2 + alpha^2)) * (z-x);
    sig = repmat((alpha^2 * rho^2) / (alpha^2 + rho^2),1,N^2);
    mu = mvnrnd(moy',sig)';
    U_MC(:,:,t+1) = reshape(mu,[N,N]);
    clear x z moy sig mu;
    
    % Show iteration counter
    waitbar(t/N_MC);
    
end
%-------------------------------------------------------------------------

t_1 = toc;
close(h);
disp('END OF THE GIBBS SAMPLING');
disp(['Execution time of the Gibbs sampling: ' num2str(t_1) ' sec']);

end
