
%-------------------------------------------------------------------------%
%                   SAMPLING THE SPLITTING VARIABLE Z                     %
%-------------------------------------------------------------------------%

function Z_new = PMYULA(Z,X,U,rho,beta,N)

%-------------------------------------------------------------------------%
% This function samples the splitting variable Z thanks to a proximal MCMC
% algorithm called P-MYULA (see Durmus et al., 2018)

    % INPUTS:
        % X,Z,U: current MCMC iterates (2D-array for X,Z and U)
        % rho: user-defined standard deviation of the variable of 
        %      interest x
        % beta: user-defined hyperparameter in p(z)
        % N: the dimension of X (2D-array)
        
    % OUTPUT:
        % Z_new: new value for Z (2D-array).
%-------------------------------------------------------------------------%

%-------------------------------------------------------------------------%
% PRE-PROCESSING
u = reshape(U,[N^2,1]);
x = reshape(X,[N^2,1]);
z = reshape(Z,[N^2,1]);
lambda_MYULA = rho^2; % as prescribed in Durmus et al.
gamma_MYULA = (rho^2)/4; % as prescribed in Durmus et al.

% 1. SAMPLE THE ZERO-MEAN GAUSSIAN VARIABLE B.
b = mvnrnd(zeros(1,N^2),ones(1,N^2))';

% 2. UPDATE THE VALUE OF Z.
    % 2.1. Compute the gradient of f(z) = (1 / (2 * rho)) ...
    % * ||z - (x - u)||_2^2.
    grad_f = (1 / rho^2) * (z - (x + u));
    
    % 2.2. Compute the proximal operator of g: prox(z)^(lambda_MYULA)_g.
    [prox_z,~] = chambolle_prox_TV_stop(Z, ...
                                        'lambda', beta*lambda_MYULA, ...
                                        'maxiter', 20);
    prox_z = reshape(prox_z,[N^2,1]);
    
    % 2.3. Compute the new value of z: z_new.
    z_new = (1 - gamma_MYULA/lambda_MYULA) * z ...
            - gamma_MYULA * grad_f ...
            + (gamma_MYULA/lambda_MYULA) * prox_z ...
            + sqrt(2 * lambda_MYULA) * b;
        
    % 2.4. Reshape z_new (1D-array) into Z_new (2D-array).
    Z_new = reshape(z_new,[N,N]);

%-------------------------------------------------------------------------%

end