function patch = cutImintoPatchRGB(hstep, wstep, noW, noH, imOrig)
% Cuts the image into patches

patch = zeros(hstep, wstep, 3, noW*noH);
count = 1;
% Cutting the patches
for i = 1:noH
    for j = 1:noW
        patch(:, :, :, count) = imOrig(1+(i-1)*hstep: i*hstep, 1+(j-1)*wstep: j*wstep, :);
        count = count +1;
    end 
end