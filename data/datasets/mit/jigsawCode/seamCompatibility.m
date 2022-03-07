function [compLR, compDU] = seamCompatibility(patch, noPatches, patchSize, overlap)

%Computing compatibilityLR matrix
compLR = zeros(noPatches, noPatches);
for i = 1:noPatches
    i
    for j = 1:noPatches
       
        EColor = sum((patch(:, patchSize - overlap + 1:patchSize, :, j) - patch(:, 1:overlap, :, i)).^2, 3);
        %Compute the mincut array        
        E = EColor;
        [cost] = seamEnergyCompute(E, 0);      
        
        compLR(i, j) = cost;
    end
end

compDU = zeros(noPatches, noPatches);
for i = 1:noPatches
    i
    for j = 1:noPatches
%         j
       
        EColor = sum((patch(patchSize - overlap + 1:patchSize,:, :, i) - patch(1:overlap,:, :, j)).^2, 3);
            
        E = EColor;
        
        %Compute the mincut array
        [cost] = seamEnergyCompute(E, 1);
        
        compDU(i, j) = cost;
    end

end

compLR = compLR/patchSize;
compDU = compDU/patchSize;