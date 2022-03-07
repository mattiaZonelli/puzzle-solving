function [patch, M_patches, N_patches] = im2patches( I, sz, overlap, patchSize )

M_patches = floor((sz(1)-overlap)/(patchSize-overlap));  % The number of patches along the vertical dimension
N_patches = floor((sz(2)-overlap)/(patchSize-overlap));  % The number of patches along the horizontal dimension
I = I(1:M_patches*(patchSize-overlap) + overlap, 1:N_patches*(patchSize-overlap) + overlap, :); % Cropping the image to fit the patches
patch = cutImintoPatchRGBOverlap(patchSize, N_patches, M_patches, overlap, I);    % Sampling the patches in a overlapping way

return