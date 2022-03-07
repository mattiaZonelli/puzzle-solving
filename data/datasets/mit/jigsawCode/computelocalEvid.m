function [local, energyVec] = computelocalEvid(evidence, patch, sig)
% This compute the local evidence p(y|x), where y = evidence, x = patches


sPatch = size(patch, 4);
local = zeros(1, sPatch);
noPixelsPatch = (size(patch, 1)*size(patch, 2));
energyVec = zeros(1, sPatch);
% tic;
for i = 1:sPatch
    energy = 0;
    for j = 1:size(evidence, 3)
        energy = energy + ((evidence(j) - sum(sum(patch(:, :, j, i)))/noPixelsPatch)^2);
    end
    energyVec(i) = energy;
    local(i) = exp(-1*energy/(2*sig^2));
end
% toc;