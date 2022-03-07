function [compDU, compLR, DUClrDist, LRClrDist] = computeCompatibilityStableColor(patch, patchColor, wstep, hstep, noPatches)
% This function computes the DU and LR compatibility matrice
% When this function is used instead of the computeCompatibility function,
% the initMRFPatchWConst should be changed.  

%load 'learntFromBerkely.mat'
%load 'derivPisSigs.mat'

filtsz = 15;
filtNo = 25;

% band length
bandLth = 3;


%Computing compatibilityLR matrix
compLR = zeros(noPatches, noPatches);
LRClrDist = zeros(noPatches, noPatches);
for i = 1:noPatches
    %Filter output
    filtOutput = zeros(1, noPatches);
    junctSeg = zeros(hstep, filtsz);
    junctSeg(:, ((filtsz+1)/2)+1 : filtsz) = patch(:, 1:(filtsz-1)/2, i);
    %meanRef = mean(mean(patchColor(:, 1:bandLth, :, i), 2), 1);
    
    
    
    %For all images, you compute the compatibility measure through the
    %image prior
    for j = 1:noPatches
        junctSeg(:, 1:(filtsz+1)/2) = patch(:, (wstep-(filtsz-1)/2):wstep, j);
        junctCol = im2col(junctSeg, [filtsz, filtsz]);
        
        % Need to change this to a gaussian scale mixture model => Refer to
        % Yair's demo MATLAB code
        xscaledSum = zeros(filtNo, 1);
        
        %Filters
        for k = 1:filtNo
            filt = ws(:, k);
            filtOutTemp = sum(junctCol.*repmat(filt, 1, size(junctCol, 2)), 1).^2;
            
            xscaled = zeros(size(filtOutTemp));
            % for different i's in Yair's paper
            for gsm = 1:8
                xscaled = xscaled + (pis(gsm)/sigs(gsm))*exp(-1*filtOutTemp/(2*sigs(gsm)));
            end
            % Summing over i in the paper
            xscaledSum(k) = sum(log(xscaled));
        end   
        % summing over k in the paper
        filtOutput(j) = sum(xscaledSum);
        
        %meanTest = mean(mean(patchColor(:, wstep-bandLth + 1:wstep, :, j), 2), 1);
%        LRClrDist(i, j) = mean(abs(meanRef - meanTest));
        LRClrDist(i, j) = sum(sum((patchColor(:, 1, :, i) - patchColor(:, wstep, :, j)).^2));
    end
    compLR(i, :) = filtOutput;
    i
end


%Computing compatibilityDU matrix
compDU = zeros(noPatches, noPatches);
DUClrDist = zeros(noPatches, noPatches);
for i = 1:noPatches
    %Filter output
    filtOutput = zeros(1, noPatches);
    junctSeg = zeros(wstep, filtsz);
    junctSeg(:, 1:(filtsz+1)/2) = patch((hstep-(filtsz-1)/2):hstep, :, i)';
%     meanRef = mean(mean(patchColor(hstep - bandLth + 1:hstep, :, :, i), 2), 1);
    
    %For all images, you compute the compatibility measure through the
    %image prior
    for j = 1:noPatches
        junctSeg(:, ((filtsz+1)/2)+1 : filtsz) = patch(1:(filtsz-1)/2, :, j)';

        junctCol = im2col(junctSeg, [filtsz, filtsz]);
        
        % Need to change this to a gaussian scale mixture model => Refer to
        % Yair's demo MATLAB code
        xscaledSum = zeros(filtNo, 1);
        for k = 1:filtNo
            filt = ws(:, k);
            filtOutTemp = sum(junctCol.*repmat(filt, 1, size(junctCol, 2)), 1).^2;
            
            xscaled = zeros(size(filtOutTemp));
            % for different i's in Yair's paper
            for gsm = 1:8
                xscaled = xscaled + (pis(gsm)/sigs(gsm))*exp(-1*filtOutTemp/(2*sigs(gsm)));
            end
            % Summing over i in the paper
            xscaledSum(k) = sum(log(xscaled));
        end   
        % summing over k in the paper
        filtOutput(j) = sum(xscaledSum);
  
%         meanTest = mean(mean(patchColor(1:bandLth, :, :, j), 2), 1);
%         DUClrDist(i, j) = mean(abs(meanRef - meanTest));

        DUClrDist(i, j) = sum(sum((patchColor(hstep, :, :, i) - patchColor(1,:, :, j)).^2));
        
    end
    compDU(i, :) = filtOutput;
    i       
end