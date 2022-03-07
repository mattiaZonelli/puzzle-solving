function [compDU, compLR, DUClrDist, LRClrDist] = compCompatibilityColor(patchColor, patchSize, noPatches)
% This function computes the DU and LR compatibility matrice
% When this function is used instead of the computeCompatibility function,
% the initMRFPatchWConst should be changed.  

LRClrDist = zeros(noPatches, noPatches);
for i = 1:noPatches
    for j = 1:noPatches    
        LRClrDist(i, j) = sum(sum((patchColor(:, 1, :, i) - patchColor(:, patchSize, :, j)).^2));
    end
    i
end


%Computing compatibilityDU matrix
DUClrDist = zeros(noPatches, noPatches);
for i = 1:noPatches
    for j = 1:noPatches
        DUClrDist(i, j) = sum(sum((patchColor(patchSize, :, :, i) - patchColor(1,:, :, j)).^2));        
    end
    i       
end

compDU = ones(noPatches, noPatches);
compLR = ones(noPatches, noPatches);