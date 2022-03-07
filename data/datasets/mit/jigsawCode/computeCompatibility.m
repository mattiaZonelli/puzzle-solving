function [compDU, compLR, flag] = computeCompatibility(patch, wstep, hstep, noPatches)
% This function computes the DU and LR compatibility matrices.

%%
%load 'learntFromBerkely.mat'
%load 'derivPisSigs.mat'

filtsz = 15;
filtNo = 25;

% for numerical stability
xscaleConst =90;

%Computing compatibilityLR matrix
compLR = zeros(noPatches, noPatches);
for i = 1:noPatches
    %Filter output
    filtOutput = zeros(1, noPatches);
    junctSeg = zeros(hstep, filtsz);
    junctSeg(:, ((filtsz+1)/2)+1 : filtsz) = patch(:, 1:(filtsz-1)/2, i);
    %For all images, you compute the compatibility measure through the
    %image prior
    for j = 1:noPatches
        junctSeg(:, 1:(filtsz+1)/2) = patch(:, (wstep-(filtsz-1)/2):wstep, j);
        junctCol = im2col(junctSeg, [filtsz, filtsz]);
        
        % Need to change this to a gaussian scale mixture model => Refer to
        % Yair's demo MATLAB code
        xscaledProd = zeros(filtNo, 1);
        
        %Filters
        for k = 1:filtNo
            filt = ws(:, k);
            filtOutTemp = sum(junctCol.*repmat(filt, 1, size(junctCol, 2)), 1).^2;
            
            xscaled = zeros(size(filtOutTemp));
            % for different i's in Yair's paper
            for gsm = 1:8
                xscaled = xscaled + (pis(gsm)/sigs(gsm))*exp(-1*filtOutTemp/(2*sigs(gsm)));
            end
            % Multiplying over i in the paper
            xscaledProd(k) = prod(xscaled/xscaleConst);
        end   
        % Multiplying over k in the paper
        filtOutput(j) = prod(xscaledProd);
    end
    compLR(i, :) = filtOutput;
    i
end

flag = 0;

if(sum(max(compLR) == 0)~= 0)
    flag = 1;
end

if(sum(max(compLR') == 0)~= 0)
    flag = 1;
end

if(flag == 1)
    return;
end

%Computing compatibilityDU matrix
compDU = zeros(noPatches, noPatches);
for i = 1:noPatches
    %Filter output
    filtOutput = zeros(1, noPatches);
    junctSeg = zeros(wstep, filtsz);
    junctSeg(:, 1:(filtsz+1)/2) = patch((hstep-(filtsz-1)/2):hstep, :, i)';
    %For all images, you compute the compatibility measure through the
    %image prior
    for j = 1:noPatches
        junctSeg(:, ((filtsz+1)/2)+1 : filtsz) = patch(1:(filtsz-1)/2, :, j)';

        junctCol = im2col(junctSeg, [filtsz, filtsz]);
        
        % Need to change this to a gaussian scale mixture model => Refer to
        % Yair's demo MATLAB code
        xscaledProd = zeros(filtNo, 1);
        for k = 1:filtNo
            filt = ws(:, k);
            filtOutTemp = sum(junctCol.*repmat(filt, 1, size(junctCol, 2)), 1).^2;
            
            xscaled = zeros(size(filtOutTemp));
            % for different i's in Yair's paper
            for gsm = 1:8
                xscaled = xscaled + (pis(gsm)/sigs(gsm))*exp(-1*filtOutTemp/(2*sigs(gsm)));
            end
            xscaledProd(k) = prod(xscaled/xscaleConst);
        end   
        filtOutput(j) = prod(xscaledProd);
    end
    compDU(i, :) = filtOutput;
    i       
end


if(sum(max(compDU) == 0)~= 0)
    flag = 1;
end

if(sum(max(compDU') == 0)~= 0)
    flag = 1;
end

if(flag == 1)
    return;
end

