function [patchLabel, patchLC, patchRC, patchTR, patchBR] = patchLabeling(patchLabel, compDU, compLR)

compUD = compDU';
compRL = compLR';
patchLC = [];
patchRC = [];
patchTR = [];
patchBR = [];

for i = 1:length(patchLabel)
    i    
    % check if there are neighbors to the right
    if(min(compRL(i, :)) ~= 0)
        patchLabel{i}.RC = 1;
        patchRC = [patchRC; i];
    else
        patchLabel{i}.RC = 0;
    end
    
    % check if there are neighbors to the left
    if(min(compLR(i, :)) ~= 0)
        patchLabel{i}.LC = 1;
        patchLC = [patchLC; i];
    else
        patchLabel{i}.LC = 0;
    end    
    
    % check if there are neighbors to the top
    if(min(compUD(i, :)) ~= 0)
        patchLabel{i}.TR = 1;
        patchTR = [patchTR; i];
    else
        patchLabel{i}.TR = 0;
    end      
    
    % check if there are neighbors to the down
    if(min(compDU(i, :)) ~= 0)
        patchLabel{i}.BR = 1;
        patchBR = [patchBR; i];    
    else
        patchLabel{i}.BR = 0;        
    end          
end

