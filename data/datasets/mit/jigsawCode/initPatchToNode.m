function [nodes, evidMatrix] = initPatchToNode(nodes, patch, evidence, noPatches, sigEvid, patchMoveFrom, patchMoveTo, patchFixOrig, patchFix, patchRemove,withEvidence, likeClr, sigClr, likeClrOn, blend)

% probabilityDist = getPropMat(kstates, sigO);
sig = sigEvid;
noH = size(evidence, 1);
noW = size(evidence, 2);
nNodes = noH*noW;
evidMatrix = zeros(nNodes, noPatches);

%%  Reordering the patch misplacements due to removed patches
for i = 1:length(patchMoveFrom)
    x = length(find(patchRemove <= patchMoveFrom(i)));
    patchMoveFrom(i) = patchMoveFrom(i) - x;
end

for i = 1:length(patchFix)
    x = length(find(patchRemove <= patchFixOrig(i)));
    patchFixOrig(i) = patchFixOrig(i) - x;
end

for i = 1:size(patchRemove, 2)
    evid = ones(1, noPatches);
    nodes{patchRemove(i)}.localEvidence = evid/sum(evid(:));
    evidMatrix(patchRemove(i), :) = evid;
end


if(likeClrOn)
    % Computing the color affinity
    clrAffinity = computeClrAffinity(likeClr, patch, sigClr);
else
    clrAffinity = ones(1, noPatches);
end


%%
% Setting the local evidence.  If withEvidence is 0, then the local
% evidence is not given to the nodes
if(withEvidence)
    iNode = 1;
    for i = 1:noH
        for j = 1:noW
            nodes{iNode}.localEvidence = computelocalEvid(evidence(i, j, :), patch, sig).*clrAffinity;
            nodes{iNode}.localEvidence = nodes{iNode}.localEvidence/sum(nodes{iNode}.localEvidence);
            evidMatrix(iNode, :) = nodes{iNode}.localEvidence/max(nodes{iNode}.localEvidence);
            iNode = iNode + 1;
        end
    end
else
    
    iNode = 1;
    for i = 1:nNodes
        nodes{i}.localEvidence =clrAffinity/sum(clrAffinity);
        evidMatrix(i, :) = nodes{iNode}.localEvidence*noPatches;
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

if(withEvidence)
    if(~blend)
        patchMoveUnique = setdiff(patchMoveFrom, patchMoveTo);
        for i = 1:size(patchMoveUnique, 2)
            nodes{patchMoveUnique(i)}.localEvidence = ones(1, noPatches)/noPatches;
            evidMatrix(patchMoveUnique(i),:) = nodes{patchMoveFrom(i)}.localEvidence/max(nodes{patchMoveFrom(i)}.localEvidence);
        end
    end
end


for i = 1:size(patchFix, 2)
    evid = zeros(1, noPatches);
    evid(patchFixOrig(i)) = 1;
    nodes{patchFix(i)}.localEvidence = evid;
    evidMatrix(patchFix(i), :) = evid;
end
