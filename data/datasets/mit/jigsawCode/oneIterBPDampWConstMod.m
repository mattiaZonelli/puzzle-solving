% Sept. 2, 2003 
% maxProductFlag = 1 means max product (MAP estimate), not sum product
% (MMSE estimate).

function [nodes, msgToFactor] = oneIterBPDampWConstMod(nodes, maxProductFlag, alpha, kStates, exclusive, sigExcl)
% Alpha is the weighting factor between the previous message and the
% current message

% Go through every link, and transmit messages near to far
% ----------------------------------------------------------------
% fprintf(1, '\nDoing belief propagation for %d nodes:\n', length(nodes)); tic;

% ----------------------------------------------------------------
for nearIdx = 1:length(nodes); 
    
    near = nodes{nearIdx};
    % message to factor
    msgToFactor = (near.localEvidence);
    
    for linkIdx = 1:near.nLinks
        
        % For local messages
        % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if(exclusive)
            messageToNode = (near.localEvidence).*(near.msgFactorToMe);
        else
            messageToNode = near.localEvidence;
        end
        
        link   = near.links{linkIdx};
        farIdx = link.farsIndex;
        for link2Idx = 1:near.nLinks 
            if (link2Idx ~= linkIdx)
                link2 = near.links{link2Idx};        
                if ~isempty(link2.farsMessageToMe)
                    % Updating the message to neighboring nodes
                    messageToNode = (messageToNode).*(link2.farsMessageToMe);                    
                end
            end; 
        end;
                
        messageToNode = messageToNode/sum(messageToNode(:));
        
        if (maxProductFlag) 
            messageToNode = max(nodes{nearIdx}.links{linkIdx}.propMat...
                .*repmat(messageToNode,size(nodes{nearIdx}.links{linkIdx}.propMat,1),1),[],2)';
        else messageToNode = (nodes{nearIdx}.links{linkIdx}.propMat *messageToNode')'; 
        end;
        
        nodes{ farIdx }.links{ link.farsLinkToMe }.fars_NEXT_MessageToMe = messageToNode;
        % - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
      
        % For message to the factor
        % - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
        if ~isempty(link.farsMessageToMe)
            % Updating the message to the factor node
            msgToFactor = (msgToFactor).*(link.farsMessageToMe);
        end
        % - - - - - - - - - - - - - - - - - - - - - - - - - - -  - - - - -
               
    end;
    
    % Updating the message to factor
    nodes{nearIdx}.msgToFactor = msgToFactor/sum(msgToFactor(:));    
end;


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

%------------------------------------------------------------------
% now *clock*, since we seem to do everything clocked these days
%------------------------------------------------------------------

% fprintf(1, 'Clocking.... '); tic

for node = 1:length(nodes); 
    for link = 1:nodes{node}.nLinks
        nodes{node}.links{link}.farsMessageToMe = ... 
            (1-alpha)*nodes{node}.links{link}.farsMessageToMe + (alpha)*nodes{node}.links{link}.fars_NEXT_MessageToMe;
    end; % links to this node  
    
    if(exclusive)
        nodes = computeMsgFactorToMeOld(nodes, msgToFactor, node, kStates, sigExcl);
    end
end; % nodes
