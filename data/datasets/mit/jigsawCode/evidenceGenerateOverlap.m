% This function generates the local evidence at each image nodes by
% computing the mean color of the patch that used to be present at that
% node
function [evidence, mask] = evidenceGenerateOverlap(im, noH, noW, patchSize, overlap)

evidenceTemp = zeros(noH, noW, size(im, 3));
mask = ones(noH, noW);

% Computing the color averages
for i = 1:noH
    istart = (i-1)*(patchSize - overlap) +1;
    for j = 1:noW
        
        jstart = (j-1)*(patchSize - overlap) + 1;
        for k = 1:size(im, 3)
            imPatch = im(istart: istart + patchSize - 1, jstart: jstart + patchSize - 1, k);
            evidenceTemp(i, j, k) = (mean(imPatch(:)));
        end         
    end 
end

evidence = evidenceTemp;