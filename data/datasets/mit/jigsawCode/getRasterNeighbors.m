% function [links] = getRasterNeighbors(iNode, N, M)
% returns a vector with the node index of each node's neighbors.
% Assumes an N x M array of nodes, 4-connected to their
% neighbors.  N rows, M columns.
% Oct. 5, 2003  wtf created.

function [links, irow, jcol] = getRasterNeighbors(iNode, N, M);

% find row and column index of node number iNode
irow = mod(iNode-1, N)+1;
jcol = ceil(iNode / N);

% find the node number indices of all the neighboring nodes to node iNode
links = [];
if irow > 1
    links = [links, iNode - 1]; 
else
    links = [links, 0];
end
if irow < N
    links = [links, iNode + 1]; 
else
    links = [links, 0];
end
if (jcol > 1)
    links = [links, iNode - N]; 
else
    links = [links, 0];
end
if jcol < M; 
    links = [links, iNode + N];
else
    links = [links, 0];
end
