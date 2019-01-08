
%-------------------------------------------------------------------------%
%                        INITIAL VARIABLES AND PARAMETERS                 %
%-------------------------------------------------------------------------%

clearvars;

rng(1); % set the seed

% load original image
X = double(imread('cameraman.tif'));
N = size(X,1);

% random observations (image inpainting (with 40% of the pixels missing)
H = rand(N)> 0.4; % 40% of the pixels missing
Y = X.* H;

% set BSNR
BSNR = 40; % SNR expressed in decibels
P_signal = var(X(:)); % signal power
sigma = sqrt((P_signal/10^(BSNR/10))); % standard deviation of the noise

% add noise
Y = H.*(Y + sigma*randn(N));

% user-defined hyperparameters
tau = 0.2*sigma^2; % regularization parameter used by Afonso et al. (2010) 
rho = 5e-3; % penalty parameter used by Afonso et al. (2010)
rho  = 3; % hyperparameter used in SPA
beta = 0.2; % regularization parameter
alpha = 1; % hyperparameter used in SPA

% number of iterations in Chambolle algorithm
TViters = 20;

% MCMC parameters
N_MC = 5000; % total number of MCMC iterations
N_bi = 200; % number of burn-in iterations

% precomputing
    % precompute the real matrix H for E-PO algorithm used within SPA
    Hmat = reshape(H,[N^2,1]);
    k2 = find(~Hmat);
    Hmat = speye(N^2);
    Hmat(k2,:) = [];
    clear k2;
    
    % precompute the real vector of observations for E-PO algorithm used 
    % within SPA
    y_signal = Hmat*reshape(X,[N^2,1]) + sigma*sprandn(N^2);
    
    % precompute the inverse of the precision matrix Q = 1/sigma^2 * H^T *
    % H + 1/rho^2 * I_N for E-PO algorithm
    % cf. Sherman-Morrison-Woodbury formula
    invQ = (rho^2) * ...
       (speye(N^2) - ((rho^2) / (sigma^2 + rho^2)) * (Hmat') * Hmat);
    invQ = @(x)  invQ*x;
    
    % precompute the dimension of the observations vector y
    M = size(y_signal,1);

% save the initial parameters
save('initial_param_SPA.mat');

% check that the file is saved
fprintf('Initial parameters saved! \n');
