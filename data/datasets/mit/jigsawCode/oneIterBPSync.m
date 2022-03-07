
function [nodes, msgToFactor] = oneIterBPSync(nodes, alpha, kStates, exclusive, sigExcl, cDU, cLR, cUD, cRL, patchDet)

undetPatch = setdiff(1:length(nodes), patchDet); 
% msgTFThresh = 1E-20;

for i = undetPatch
    % Getting the messages from the neighboring nodes
    if(nodes{i}.NL ~= 0)
        nodes{i}.msgTable(1, :) = nodes{nodes{i}.NL}.prevMTR;
    end
    
    if(nodes{i}.NR ~= 0)
        nodes{i}.msgTable(2, :) = nodes{nodes{i}.NR}.prevMTL;
    end
    
	if(nodes{i}.NT ~= 0)
        nodes{i}.msgTable(3, :) = nodes{nodes{i}.NT}.prevMTD;
    end
    
    if(nodes{i}.ND ~= 0)
        nodes{i}.msgTable(4, :) = nodes{nodes{i}.ND}.prevMTT;
    end
    
    % Sending the messages to the neighboring nodes
    % to left node
    if(exclusive)
        msgAgg =  prod(nodes{i}.msgTable(2:4, :), 1).*nodes{i}.msgFactorToMe.*nodes{i}.localEvidence;
    else
        msgAgg =  prod(nodes{i}.msgTable(2:4, :), 1).*nodes{i}.localEvidence;
    end
    
    if(nodes{i}.NL ~= 0)
        nodes{i}.MTL = (cLR'*msgAgg')';
        nodes{i}.MTL = nodes{i}.MTL/sum(nodes{i}.MTL);
    end
    
    % to right node
    if(exclusive)
        msgAgg =  prod(nodes{i}.msgTable([1, 3:4], :), 1).*nodes{i}.msgFactorToMe.*nodes{i}.localEvidence;
    else
        msgAgg =  prod(nodes{i}.msgTable([1, 3:4], :), 1).*nodes{i}.localEvidence;
    end
    
    if(nodes{i}.NR ~= 0)
        nodes{i}.MTR = (cRL'*msgAgg')';
        nodes{i}.MTR = nodes{i}.MTR/sum(nodes{i}.MTR);
    end
    
    % to top node
    if(exclusive)
        msgAgg =  prod(nodes{i}.msgTable([1:2, 4], :), 1).*nodes{i}.msgFactorToMe.*nodes{i}.localEvidence;
    else
        msgAgg =  prod(nodes{i}.msgTable([1:2, 4], :), 1).*nodes{i}.localEvidence;
    end
    if(nodes{i}.NT ~= 0)
        nodes{i}.MTT = (cUD'*msgAgg')';
        nodes{i}.MTT = nodes{i}.MTT/sum(nodes{i}.MTT);
    end
    
    % to down node
    if(exclusive)
        msgAgg =  prod(nodes{i}.msgTable([1:3], :), 1).*nodes{i}.msgFactorToMe.*nodes{i}.localEvidence;
    else
        msgAgg =  prod(nodes{i}.msgTable([1:3], :), 1).*nodes{i}.localEvidence;
    end
    
    if(nodes{i}.ND ~= 0)
        nodes{i}.MTD = (cDU'*msgAgg')';
        nodes{i}.MTD = nodes{i}.MTD/sum(nodes{i}.MTD);
    end
    
    msgAgg =  prod(nodes{i}.msgTable, 1).*nodes{i}.localEvidence;
    nodes{i}.msgToFactor = msgAgg/sum(msgAgg(:));%.*(msgAgg > msgTFThresh);    
%    nodes{i}.msgToFactor = nodes{i}.msgToFactor/sum(nodes{i}.msgToFactor);
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

%     nodes{i}.prevMTL = alpha*nodes{i}.prevMTL + (1- alpha)*nodes{i}.MTL;
%     nodes{i}.prevMTR = alpha*nodes{i}.prevMTR + (1- alpha)*nodes{i}.MTR;
%     nodes{i}.prevMTT = alpha*nodes{i}.prevMTT + (1- alpha)*nodes{i}.MTT;
%     nodes{i}.prevMTD = alpha*nodes{i}.prevMTD + (1- alpha)*nodes{i}.MTD;

    nodes{i}.prevMTL = ((nodes{i}.prevMTL).^(alpha)).*((nodes{i}.MTL).^(1-alpha));
    nodes{i}.prevMTR = ((nodes{i}.prevMTR).^(alpha)).*((nodes{i}.MTR).^(1-alpha));
    nodes{i}.prevMTT = ((nodes{i}.prevMTT).^(alpha)).*((nodes{i}.MTT).^(1-alpha));
    nodes{i}.prevMTD = ((nodes{i}.prevMTD).^(alpha)).*((nodes{i}.MTD).^(1-alpha));

    
    if(exclusive)
        nodes = computeMsgFactorToMe(nodes, msgToFactor, i, kStates, sigExcl);
    end   
end