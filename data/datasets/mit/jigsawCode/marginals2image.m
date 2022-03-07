function[im] = marginals2image(nodes,M, N)

% function[im, P] = marginals2image(nodes,M, N, noPatches)

iNode = 1;
im = zeros(M, N);
for i = 1:M
    for j = 1:N
        imax = find(nodes{iNode}.marginal == max(nodes{iNode}.marginal));
%         imax = find(nodes{iNode}.localEvidence == max(nodes{iNode}.localEvidence));
%         [i, j]
        if(isempty(imax))
            [i,j]
            im = [];
            return;

            im(i,j) = 16;
        else
            im(i,j) = imax(1);
        end
        iNode = iNode + 1;
    end
end
