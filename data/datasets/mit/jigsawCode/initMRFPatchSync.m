function [nodes] = initMRFPatchSync(M_Nodes, N_Nodes, kStates, noPatches,exclusive, patchDet, cDU, cLR, cUD, cRL, patch, evidence, sigEvid, ...
    patchMoveFrom, patchMoveTo, patchFixOrig, patchFix, patchRemove,withEvidence, likeClr, sigClr, likeClrOn, blend, randInit)

%% - Now init the nodes and the bVectors:
%% -----------------------------------------------

nNodes = M_Nodes*N_Nodes;

nodes = cell(nNodes, 1);
[nodes] = initPatchToNode(nodes, patch, evidence, noPatches, ...
        sigEvid, patchMoveFrom, patchMoveTo, patchFixOrig, patchFix, patchRemove,withEvidence, likeClr, sigClr, likeClrOn, blend);


for n = 1:nNodes        
    % Message table 
    nodes{n}.msgTable = ones(4, noPatches);
    
    % This is first to find which are the neighboring nodes for the node of interest 
    links = getRasterNeighbors(n, N_Nodes, M_Nodes);
          
          
    % Node numbers in the neighborhood
    nodes{n}.NL = links(1);
    nodes{n}.NR = links(2);
    nodes{n}.NT = links(3);
    nodes{n}.ND = links(4);

    curr = nodes{n};  
    msgCollectedT = curr.localEvidence;  
    msgCollected = msgCollectedT/sum(msgCollectedT(:));
       
    curr.prevMTL = ones(1, kStates)/kStates;
    curr.prevMTR = ones(1, kStates)/kStates;
    curr.prevMTT = ones(1, kStates)/kStates;
    curr.prevMTD = ones(1, kStates)/kStates;
    
    % Initializing the messages
    
    if(nodes{n}.NL ~= 0)
        if(randInit)
            curr.prevMTL = rand(1, kStates);
        else            
            curr.prevMTL = (cLR'*msgCollected')';       
        end
    end

    if(nodes{n}.NR ~= 0)
        if(randInit)
            curr.prevMTR = rand(1, kStates);
        else  
            curr.prevMTR = (cRL'*msgCollected')';    
        end
    end

    if(nodes{n}.NT ~= 0)
        if(randInit)
            curr.prevMTT = rand(1, kStates);
        else         
            curr.prevMTT = (cUD'*msgCollected')';  
        end
    end

    if(nodes{n}.ND ~= 0)
        if(randInit)
            curr.prevMTT = rand(1, kStates);
        else      
            curr.prevMTD = (cDU'*msgCollected')';
        end
    end

    

        
    curr.MTL = ones(1, kStates)/kStates;
    curr.MTR = ones(1, kStates)/kStates;
    curr.MTT = ones(1, kStates)/kStates;
    curr.MTD = ones(1, kStates)/kStates;    
        
    nodes{n} = curr;
    
    if(exclusive)
        nodes{n}.msgFactorToMe = ones(1, kStates);
        nodes{n}.msgToFactor = ones(1, kStates);
    end
    
    nodes{n}.marginal = ones(1, kStates)/kStates;    
     
    nodes{n}.activeInd = [1:noPatches];
    nodes{n}.prevInd = [1:noPatches];
end

for n = patchDet
    nodes{n}.marginal = nodes{n}.localEvidence;    
    nodes{n}.MTL = nodes{n}.prevMTL;
    nodes{n}.MTR = nodes{n}.prevMTR;
    nodes{n}.MTT = nodes{n}.prevMTT;
    nodes{n}.MTD = nodes{n}.prevMTD;   
    
    nodes{n}.msgToFactor = nodes{n}.localEvidence;
end
