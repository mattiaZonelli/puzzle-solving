function [patchRaster] = patchRas(patch)
%This function rasterizes the patches into a matrix form
% input = patch : sx x sy x a x noPatches matrix; if a == 1, gray
% output = patchRaster: a 2D matrix where each column is a rasterized vector

[sx, sy, a, noPatches] = size(patch);
patchRaster = zeros(sx*sy*a, noPatches);

for i = 1:noPatches
    tempPatch = patch(:, :, :, i);
    patchRaster(:, i) = tempPatch(:);
end
