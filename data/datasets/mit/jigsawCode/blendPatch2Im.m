function imOut = blendPatch2Im(hstep, wstep, noH, noW, patch, nodeReconst)

imOut = zeros(noH*hstep, noW*wstep, 3);
imNull = ones(hstep, wstep, 3);
% Cutting the patches
for i = 1:noH
    for j = 1:noW
%         if(nodeReconst(i, j) ~= 1)
            imOut(1+(i-1)*hstep: i*hstep, 1+(j-1)*wstep: j*wstep, :) = patch(:, :, :, nodeReconst(i, j));
%         else
%             imOut(1+(i-1)*hstep: i*hstep, 1+(j-1)*wstep: j*wstep, :) = imNull;
%             'Note that the output image is nullified!!!!!!!!!!!!!!!!!!!!!!!!!!'
%         end
    end 
end