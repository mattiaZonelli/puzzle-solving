function patch = cutImintoPatch(hstep, wstep, noW, noH, imOrig)
% Cuts the image into patches

patch = zeros(hstep, wstep, noW*noH);
count = 1;
% Cutting the patches
for i = 1:noH
    for j = 1:noW
        patch(:, :, count) = imOrig(1+(i-1)*hstep: i*hstep, 1+(j-1)*wstep: j*wstep);
        count = count +1;
    end 
end