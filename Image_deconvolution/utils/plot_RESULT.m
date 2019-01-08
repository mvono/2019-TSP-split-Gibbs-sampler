function plot_RESULT(y,refl,X_MC,Z_MC,U_MC,N_bi,N_MC,N)

close all

% Dock the generated figures
set(0,'DefaultFigureWindowStyle','docked');

% 1. PLOT ORIGINAL, OBSERVATIONS AND ESTIMATES

    % Plot the original image
    figure(1);
    imagesc(refl,[0 255]);
    axis equal; axis off;colormap('gray');
    title('Original image');

    % Plot the noisy observation
    figure(2);
    imshow(y, [0 255]);
    axis equal; axis off;colormap('gray');
    title('Blurred and noisy observation');
 
    % Plot the MMSE of x
    figure(3);
    imagesc(mean(X_MC(:,:,N_bi:end),3),[0 255]);
    axis equal; axis off;colormap('gray');
    title('MMSE estimate of x');
    
    % Plot the MMSE of z
    figure(4);
    imagesc(mean(Z_MC(:,:,N_bi:end),3),[0 255]);
    axis equal; axis off;colormap('gray');
    title('MMSE estimate of z');
    
    % Plot the MMSE of u
    figure(5);
    imagesc(mean(U_MC(:,:,N_bi:end),3));colorbar;
    axis equal; axis off;colormap('gray');
    title('MMSE estimate of u');
    
    
% 2. PLOT THE 90% CREDIBILITY INTERVALS
CI_90 = zeros(N,N);
for i = 1:N
    for j = 1:N
        arr = reshape(X_MC(i,j,N_bi:end),[N_MC-N_bi+1,1]);
        quant_5 = quantile(arr,0.05);
        quant_95 = quantile(arr,0.95);
        CI_90(i,j) = abs(quant_95 - quant_5);
    end
end

figure(6);
imagesc(CI_90);
axis equal; axis off;colormap(flipud(gray));colorbar;
title('90% credibility intervals');

end