function patch = cutImintoPatchRGBOverlap(patchSize, N_patches, M_patches, overlap, imOrig)
% Cuts the image into patches in an overlapping fashion

patch = zeros(patchSize, patchSize, size(imOrig, 3), N_patches*M_patches);
count = 1;
% Cutting the patches
for i = 1:M_patches
    istart = (i-1)*(patchSize - overlap) +1;
    
    for j = 1:N_patches
        jstart = (j-1)*(patchSize - overlap) + 1;
              
        patch(:, :, :, count) = imOrig(istart: istart + patchSize - 1, jstart: jstart + patchSize - 1, :);
        count = count +1;
    end
end