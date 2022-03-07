
function [nodes, msgToFactor] = oneIterBP4Boundary(nodes, alpha, kStates, exclusive, sigExcl, cLR, cRL)

lenNodes = length(nodes);

for i = 1:lenNodes
    % Getting the messages from the neighboring nodes
    if(nodes{i}.NL ~= 0)
        nodes{i}.msgTable(1, :) = nodes{nodes{i}.NL}.prevMTR;
    end
    
    if(nodes{i}.NR ~= 0)
        nodes{i}.msgTable(2, :) = nodes{nodes{i}.NR}.prevMTL;
    end
    
    % Sending the messages to the neighboring nodes to left node
    if(exclusive)
        msgAgg =  (nodes{i}.msgTable(2, :)).*nodes{i}.msgFactorToMe.*nodes{i}.localEvidence;
    else
        msgAgg =  (nodes{i}.msgTable(2, :)).*nodes{i}.localEvidence;
    end
    
    if(nodes{i}.NL ~= 0)
        nodes{i}.MTL = (cLR'*msgAgg')';
        nodes{i}.MTL = nodes{i}.MTL/sum(nodes{i}.MTL);
    end
    
    % to right node
    if(exclusive)
        msgAgg =  (nodes{i}.msgTable(1, :)).*nodes{i}.msgFactorToMe.*nodes{i}.localEvidence;
    else
        msgAgg =  (nodes{i}.msgTable(1, :)).*nodes{i}.localEvidence;
    end
    
    if(nodes{i}.NR ~= 0)
        nodes{i}.MTR = (cRL'*msgAgg')';
        nodes{i}.MTR = nodes{i}.MTR/sum(nodes{i}.MTR);
    end
      
    msgAgg =  prod(nodes{i}.msgTable, 1).*nodes{i}.localEvidence;
    nodes{i}.msgToFactor = msgAgg/sum(msgAgg(:));
end


% ------------------------------------------------------------
% Computing the product of all messages to the factor
% ------------------------------------------------------------
if(exclusive)
    msgToFactor = ones(length(nodes), kStates);
    for j = 1:size(nodes, 1)
        msgToFactor(j, :) = (ones(1, kStates) - nodes{j}.msgToFactor);
    end
else
    msgToFactor = 0;
end


for i = undetPatch
    
    nodes{i}.prevMTL = ((nodes{i}.prevMTL).^(alpha)).*((nodes{i}.MTL).^(1-alpha));
    nodes{i}.prevMTR = ((nodes{i}.prevMTR).^(alpha)).*((nodes{i}.MTR).^(1-alpha));
    
    
    if(exclusive)
        nodes = computeMsgFactorToMe(nodes, msgToFactor, i, kStates, sigExcl);
    end   
end