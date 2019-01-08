
%-------------------------------------------------------------------------%
%             E-PO ALGORITHM TO SAMPLE THE VARIABLE X                     %
%-------------------------------------------------------------------------%

function x = EPO(y,H,sigma,U,Z,rho,N,M,invQ)

%-------------------------------------------------------------------------%
% This function computes the E-PO algorithm as described in the paper of C.
% Gilavert et al., 2015. This algorithm deals with the exact resolution 
% case of the linear system Q*x = eta and with a guaranteed convergence to 
% the target distribution.

    % INPUTS:
        % y: noisy observation (1D array).
        % H: direct operator in the linear inverse problem y = H*x + n.
        % sigma: user-defined standard deviation of the noise.
        % U,Z,delta: current MCMC iterates of the other variables.
        % rho: user-defined standard deviation of the variable of 
        %      interest x.
        % N,M: respectively, the dimension of X (2D-array) and y
        % (1D-array).
        % invQ: pre-computed covariance matrix involved in the posterior
        % distribution of the variable of interest x.
        
    % OUTPUT:
        % x: sample from the posterior distribution of x (2D-array).
%-------------------------------------------------------------------------%

%-------------------------------------------------------------------------%
% 1. Sample eta from N(Q*mu,Q)
    % 1.1. Sample eta_y from N(y,sigma^2*I_M)
    diagY = diag((sigma^2)*speye(M));
    eta_y = mvnrnd(y, diagY')';
    clear y diagY;

    % 1.2. Sample eta_x from N(z-delta*d,rho^2*I_N)
    u = reshape(U,[1,N^2]);
    z = reshape(Z,[1,N^2]);
    diagX = diag((rho^2)*speye(N^2));
    eta_x = mvnrnd(z-u, diagX')';
    clear d z diagX;

    % 3. Set eta
    eta_aux = (1 / sigma^2) * (H')* eta_y + (1 / rho^2) * eta_x;
    eta_aux = reshape(eta_aux,[N^2,1]);
    clear eta_y eta_x;
%-------------------------------------------------------------------------%

%-------------------------------------------------------------------------%
% 2. Compute an exact solution x_new of Q*x = eta <=> x = invQ*eta
x = invQ(eta_aux);
x = reshape(x,[N,N]);
%-------------------------------------------------------------------------%

end