function [boundaryLabel] = boundaryPatchAssgn(patchLabel, cRL, cLR, lenNodes)

% Declare nodes
nodes = cell(1, lenNodes);
kStates = length(patchLabel);

exclusive = 1;
nIter = 2*lenNodes + 1;

%% Initialize the nodes
for n = 1:lenNodes
    % Message table 
    nodes{n}.msgTable = ones(2, noPatches);
              
    % Node numbers in the neighborhood
    if(n == 1)
        nodes{n}.NL = 0;
    else
        nodes{n}.NL = n-1;
    end

    if(n == lenNodes)
        nodes{n}.NR = 0;
    else
        nodes{n}.NR = n+1;
    end             
    
    curr.prevMTL = ones(1, kStates)/kStates;
    curr.prevMTR = ones(1, kStates)/kStates;
               
    curr.MTL = ones(1, kStates)/kStates;
    curr.MTR = ones(1, kStates)/kStates;

    if(exclusive)
        nodes{n}.msgFactorToMe = ones(1, kStates);
        nodes{n}.msgToFactor = ones(1, kStates);
    end
     
    nodes{n} = curr;        
    nodes{n}.marginal = ones(1, kStates)/kStates;    
end

%% Run BP
tic;
for i = 1:nIter
    [nodes] = oneIterBP4Boundary(nodes, alpha, kStates, exclusive, sigExcl, cLR, cRL);
end
toc;

nodes = computeBeliefs(nodes,[], kStates);
[nodeReconst] = marginals2image(nodes, 1, lenNodes);  %, noPatches - noRemove + noAdd)
