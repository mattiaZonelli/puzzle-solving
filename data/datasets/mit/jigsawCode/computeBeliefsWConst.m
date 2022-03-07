% input:  a set of nodes
% output: that same set of nodes, but with the beliefs set appropriately.
% Sept. 3, 2003 
%
% Updated
% Oct 23 2007

function nodes = computeBeliefsWConst(nodes, exclusive)

for nearIdx = 1:length(nodes);     
    near = nodes{nearIdx};
    
    % Multiplying the message from the factor node
    if(exclusive)        
        belief = (near.localEvidence).*(near.msgFactorToMe);
    else
        belief = near.localEvidence;
    end
    for linkIdx = 1:near.nLinks
        link   = near.links{linkIdx};      
        if ~isempty(link.farsMessageToMe)
            belief = (belief).*(link.farsMessageToMe);
        end
    end;
    nodes{nearIdx}.marginal = belief/sum(belief(:));
end;

