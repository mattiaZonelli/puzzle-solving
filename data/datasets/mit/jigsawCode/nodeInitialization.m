function [patchLabel] = nodeInitialization(noPatches)

nodes = cell(1, noPatches);

for i = 1:noPatches
    nodes{i}.TR = 0;
    nodes{i}.BR = 0;
    nodes{i}.LC = 0;
    nodes{i}.RC = 0;   
end

patchLabel = nodes;