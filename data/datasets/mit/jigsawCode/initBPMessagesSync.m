% Initializing the messages in the node to be used for synchronous message
% passing scheme.
function [nodes] = initBPMessagesSync(nodes, kStates, exclusive, patchDet, cDU, cLR, cUD, cRL)

nNodes = length(nodes);
% Randomly initializing messages
for n = 1:nNodes
    
    curr = nodes{n};  
    msgCollectedT = curr.localEvidence;  
    msgCollected = msgCollectedT/sum(msgCollectedT(:));
    
    %% Sending out messages to nodes with lower priority    
    % Node to the left
     
    curr.prevMTL = (cLR'*msgCollected')';        
    curr.prevMTR = (cRL'*msgCollected')';    
    curr.prevMTT = (cUD'*msgCollected')';    
    curr.prevMTD = (cDU'*msgCollected')';
        
    curr.MTL = ones(1, kStates);
    curr.MTR = ones(1, kStates);
    curr.MTT = ones(1, kStates);
    curr.MTD = ones(1, kStates);    
        
    nodes{n} = curr;
    
    if(exclusive)
        nodes{n}.msgFactorToMe = ones(1, kStates);
        nodes{n}.msgToFactor = ones(1, kStates);
    end
    
    nodes{n}.marginal = ones(1, kStates)/kStates;
end

for n = patchDet
    nodes{n}.marginal = nodes{n}.localEvidence;    
end