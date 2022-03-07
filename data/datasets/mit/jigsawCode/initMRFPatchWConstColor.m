
function [nodes, cDU, cLR, cUD, cRL] = initMRFPatchWConstColor(M_Nodes, N_Nodes, K_states, compDU, compLR, logComp, noPatches, DUClrDist, LRClrDist)
fprintf(1, 'Entering initMRFPatchWConst \n'); tic;                    %%

nNodes = N_Nodes * M_Nodes;

% load '4_compDUYair.mat'
% load '4_compLRYair.mat'
warning off all;
% % 
% compDU = compDU.*(ones(noPatches) - eye(noPatches));
% compLR = compLR.*(ones(noPatches) - eye(noPatches));

cDUTemp = compDU;
cUDTemp = compDU';
cLRTemp = compLR;
cRLTemp = compLR';

cLR = zeros(size(compLR));
cRL = zeros(size(compLR));
cDU = zeros(size(compDU));
cUD = zeros(size(compDU));

sigColor = 0.2;


if(logComp == 0)
    %normalizing Compatibility Matrices
    for i = 1:noPatches
        cLR(i, :) = (cLRTemp(i, :))/sum(cLRTemp(i, :));
        cRL(i, :) = (cRLTemp(i, :))/sum(cRLTemp(i, :));
        cDU(i, :) = (cDUTemp(i, :))/sum(cDUTemp(i, :));
        cUD(i, :) = (cUDTemp(i, :))/sum(cUDTemp(i, :));
    end
else
    for i = 1:noPatches
%         cLRTemp(i, :) = exp(cLRTemp(i, :) - repmat(max(cLRTemp(:)), size(cLRTemp(i, :))));
%         cRLTemp(i, :) = exp(cRLTemp(i, :) - repmat(max(cRLTemp(:)), size(cRLTemp(i, :))));       
%         cDUTemp(i, :) = exp(cDUTemp(i, :) - repmat(max(cDUTemp(:)), size(cDUTemp(i, :))));
%         cUDTemp(i, :) = exp(cUDTemp(i, :) - repmat(max(cUDTemp(:)), size(cUDTemp(i, :))));
        

        cLRTemp(i, :) = exp(cLRTemp(i, :) - repmat(max(cLRTemp(i, :)), size(cLRTemp(i, :))));
        cRLTemp(i, :) = exp(cRLTemp(i, :) - repmat(max(cRLTemp(i, :)), size(cRLTemp(i, :))));       
        cDUTemp(i, :) = exp(cDUTemp(i, :) - repmat(max(cDUTemp(i, :)), size(cDUTemp(i, :))));
        cUDTemp(i, :) = exp(cUDTemp(i, :) - repmat(max(cUDTemp(i, :)), size(cUDTemp(i, :))));
        


        cLR(i, :) = (cLRTemp(i, :))/sum(cLRTemp(i, :));
        cRL(i, :) = (cRLTemp(i, :))/sum(cRLTemp(i, :));
        cDU(i, :) = (cDUTemp(i, :))/sum(cDUTemp(i, :));
        cUD(i, :) = (cUDTemp(i, :))/sum(cUDTemp(i, :));
        
    end
    
    %% Color information
    cDUColor = exp(-1*DUClrDist/(sigColor.^2));
    cLRColor = exp(-1*LRClrDist/(sigColor.^2));
    cUDColor = exp(-1*DUClrDist'/(sigColor.^2));
    cRLColor = exp(-1*LRClrDist'/(sigColor.^2));
    
    % CHANGE!
    
%     cDU = cDUTemp.*cDUColor;
%     cLR = cLRTemp.*cLRColor;
%     cUD = cUDTemp.*cUDColor;
%     cRL = cRLTemp.*cRLColor;
    
    cDU = cDU.*cDUColor;
    cLR = cLR.*cLRColor;
    cUD = cUD.*cUDColor;
    cRL = cRL.*cRLColor;
    
    cDU = cDU.*(ones(noPatches) - eye(noPatches));
    cLR = cLR.*(ones(noPatches) - eye(noPatches));
    cUD = cUD.*(ones(noPatches) - eye(noPatches));
    cRL = cRL.*(ones(noPatches) - eye(noPatches));
    
    for i = 1:noPatches
        cLR(i, :) = cLR(i, :) / sum(cLR(i, :));
        cRL(i, :) = (cRL(i, :))/sum(cRL(i, :));    
        cDU(i, :) = (cDU(i, :))/sum(cDU(i, :));   
        cUD(i, :) = (cUD(i, :))/sum(cUD(i, :));
    end
end




%% - Now init the nodes and the bVectors:
%% -----------------------------------------------

% nNodes +
nodes = cell(nNodes, 1);
for n = 1:nNodes
    % initialize belief to equal uncertainty.
    nodes{n}.marginal = ones(K_states,1) / K_states;
    % This is first to find which are the neighboring nodes for the node of interest 
    links = getRasterNeighbors(n, N_Nodes, M_Nodes);

    %count is introduced here not to make any link to 0
    count = 1;
    for w = 1:4
        if(links(w) ~= 0)
            nodes{n}.links{count} = cell(1, 1);
            nodes{n}.links{count}.farsIndex = links(w);
            % for generality, specify a propMat for each link and direction
            % when w = 1, then the node we are attaching is the left node.
            % When w = 2, then the node we are attaching is the right node.
            % When w = 3, then the node we are attaching is the top node.
            % When w = 4, then the node we are attaching is the bottom node.
            if w == 1
                nodes{n}.links{count}.propMat = cRL;
            elseif w == 2
                nodes{n}.links{count}.propMat = cLR;
            elseif w == 3
                nodes{n}.links{count}.propMat = cDU;
            else
                nodes{n}.links{count}.propMat = cUD;
            end
            count = count + 1;
        end
    end
    
    % Total number of links for that node
    nodes{n}.nLinks = count-1;
end

%------------------------------------------------------------------
% now go through every link, and set the links back and 
% check that each link out has exactly one link back.
%------------------------------------------------------------------

fprintf(1,'Checking bidirectionality ... \n');

for n = 1:nNodes
    
    for li = 1:(nodes{n}.nLinks)
        farIdx = nodes{n}.links{li}.farsIndex;
        nhits = 0;
        for farslink = 1:nodes{farIdx}.nLinks
            if (nodes{farIdx}.links{farslink}.farsIndex == n)
                nodes{n}.links{li}.farsLinkToMe = farslink;
                nhits = nhits + 1;
            end
        end
        if (nhits~= 1)
            keyboard
            error('nhits <> 0 in initNodes');
        end
    end
end

fprintf(1,'Done initting nodes: %f secs\n',toc);