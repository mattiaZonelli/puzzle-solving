% This function computes the marginal distribution of the nodes

function nodes = computeBeliefs(nodes, patchDet, noPatches)

undetPatch = setdiff(1:length(nodes), patchDet); 

for i = undetPatch

    msgAgg = nodes{i}.localEvidence;
    
    
    % Getting the messages from the neighboring nodes
    if(nodes{i}.NL ~= 0)
        msgAgg = msgAgg.*nodes{nodes{i}.NL}.prevMTR;
    end
    
    if(nodes{i}.NR ~= 0)
        msgAgg = msgAgg.*nodes{nodes{i}.NR}.prevMTL;
    end
    
	if(nodes{i}.NT ~= 0)
        msgAgg = msgAgg.*nodes{nodes{i}.NT}.prevMTD;
    end
    
    if(nodes{i}.ND ~= 0)
        msgAgg = msgAgg.*nodes{nodes{i}.ND}.prevMTT;
    end

    msgTemp = zeros(1, noPatches);
    msgTemp(nodes{i}.activeInd) = msgAgg;
    nodes{i}.marginal = msgTemp/sum(msgTemp(:));
    
end