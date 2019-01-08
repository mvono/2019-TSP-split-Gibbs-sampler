function [ISNR,mse] = ISNR(X,Y,X_MC,N_bi) 

mse = norm(X - mean(X_MC(:, :, N_bi:end), 3), 'fro')^2 / numel(X);
ISNR = 10 * log10 (sum((Y(:) - X(:)).^2) ./ (mse * numel(X)));

end