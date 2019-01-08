
%-------------------------------------------------------------------------%
%                        INITIAL VARIABLES AND PARAMETERS                 %
%-------------------------------------------------------------------------%

clearvars;

rng(1); % set the seed

% 1.1. Load original 512 x 512 image 
refl =  double(imread('lena.bmp'));

% 1.2. Define the regularization 
psf = [[0,-1,0];[-1,4,-1];[0,-1,0]];
[FL,FLC,F2L,~] = HXconv(refl,psf,'Hx');
delta = 1e-1;
FL = -FL + delta;
FLC = -FLC + delta;
F2L = abs(FL).^2;

% 1.3. Define the blurring kernel and its associated Fourier matrices 
B = fspecial('gaussian',39,4);
[FB,FBC,F2B,Bx] = HXconv(refl,B,'Hx');

% 1.4. Apply the blurring operator on the original image
N = numel(refl);
Ni = sqrt(N);
beta = 0.35;
kappa1 = 13;
kappa2 = 40;
D = kappa2 * binornd(1,beta,[Ni Ni]);
D(D==0) = kappa1;
y = Bx + D .* randn(Ni,Ni);

% 1.5. Define the parameters of SPA
rho = 20;
alpha = 1;

% 1.6. Define MCMC parameters
N_MC = 1000; % total number of MCMC iterations
N_bi = 200; % number of burn-in iterations

% 1.7. Other parameters and precomputing
gamma = 6e-3; % regularization parameter (fixed here)
D = D.^(-2); % precision matrix associated to the likelihood
mu1 = 0.99 / max(D(:)); % parameter used in AuxV1 method embedded in SPA
N = size(y,1);

% save the initial parameters
save('initial_param_SPA.mat','D','mu1','FB','F2B','rho','alpha',...
                             'y','FBC','gamma','F2L','N','N_MC','N_bi',...
                             'refl');

% check that the file is saved
fprintf('Initial parameters saved! \n');
