% [nodes] = initLocalEvidence(nodes, im, p);
% Given an input binary im of the appropriate resolution, and a
% probability, p, that each pixel is flipped, initialize the local evidence
% vector coming into each node.
% patch = an array of patches
% evidence = low res version of the input image
% noPatches = number of patches
% sigEvid = sigma for the evidence term
% patchMoveFrom = a vector of patch number that are to be moved from...
% patchMoveTo = a vector of patch number that are to be moved to...
% patchFix = a vector of patch number that are to be fixed...
% withEvidence = flag for using the local evidence 


function [nodes, evidMatrix, energyMat] = initPatchToNodeHis(nodes, patch, evidence, probMap, patchLabel, noPatches, sigEvid, patchFix, patchMoveFrom, patchMoveTo)

sig = sigEvid;
noH = size(evidence, 1);
noW = size(evidence, 2);
nNodes = noH*noW;
evidMatrix = zeros(nNodes, noPatches);
patchLabelT = patchLabel';
energyMat = zeros(noH*noW, noPatches);


% Local evidence & prior
iNode = 1;
for i = 1:noH
    for j = 1:noW
        [nodes{iNode}.localEvidence, energyVec] = computelocalEvid(evidence(i, j, :), patch, sig);%.*squeeze(probMap(i, j, patchLabelT(:)))';
        
        nodes{iNode}.localEvidence = nodes{iNode}.localEvidence.*squeeze(probMap(i, j, patchLabelT(:)))';
        nodes{iNode}.localEvidence = nodes{iNode}.localEvidence/sum(nodes{iNode}.localEvidence);
        evidMatrix(iNode, :) = nodes{iNode}.localEvidence;
        energyMat(:, iNode) = energyVec';
        iNode = iNode + 1;   
       
    end
end


% Selected patches are dedicated to certain nodes, and the nodes that the
% selected patches are taken from are given uniform local evidence.
for i = 1:size(patchMoveFrom, 2)
    evid = zeros(1, noPatches);
    evid(patchMoveFrom(i)) = 1;
    nodes{patchMoveTo(i)}.localEvidence = evid;
    evidMatrix(patchMoveTo(i), :) = evid;
end

patchMoveUnique = setdiff(patchMoveFrom, patchMoveTo);
for i = 1:size(patchMoveUnique, 2)
    nodes{patchMoveUnique(i)}.localEvidence = ones(1, noPatches)/noPatches;
    evidMatrix(patchMoveUnique(i),:) = nodes{patchMoveFrom(i)}.localEvidence/max(nodes{patchMoveFrom(i)}.localEvidence);
end

for i = 1:size(patchFix, 2)
    evid = zeros(1, noPatches);
    evid(patchFix(i)) = 1;
    nodes{patchFix(i)}.localEvidence = evid;
    nodes{patchFix(i)}.marginal = evid;
    evidMatrix(patchFix(i), :) = evid;
end