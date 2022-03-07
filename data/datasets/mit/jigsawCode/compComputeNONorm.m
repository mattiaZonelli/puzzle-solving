function [cDU, cLR, cUD, cRL] = compComputeNONorm(M_Nodes, N_Nodes, DUClrDist, LRClrDist, noPatches, kThresh)

warning off all;

cDUTemp = DUClrDist;
cUDTemp = DUClrDist';
cLRTemp = LRClrDist;
cRLTemp = LRClrDist';

sigFac = 1;

for i = 1:noPatches

%     i
    if(mod(i, N_Nodes) == 1)
        [cLRTempSort, cLRSortInd] = sort(cLRTemp(i, :), 'ascend');
        cLRSortInd(1:kThresh) = [];
        minCLRTemp = cLRTempSort(1);
        indMin = find(cLRTempSort == minCLRTemp);
        cLRTempSort(indMin) = [];
        sigComp = sigFac*sqrt(cLRTempSort(1) - minCLRTemp);                
        cLRTemp(i, :) = exp(-1*(cLRTemp(i, :) - repmat(min(cLRTemp(i, :)), size(cLRTemp(i, :))))/sigComp^2);
        cLRTemp(i, cLRSortInd) = 0;%0.001*min(cLRTemp(i, :));
    else
        [cLRTempSort, cLRSortInd] = sort(cLRTemp(i, :), 'ascend');
        cLRSortInd(1:kThresh) = [];
        minCLRTemp = cLRTempSort(1);
        indMin = find(cLRTempSort == minCLRTemp);
        cLRTempSort(indMin) = [];
        sigComp = sqrt(cLRTempSort(1) - minCLRTemp);                
        cLRTemp(i, :) = exp(-1*(cLRTemp(i, :) - repmat(min(cLRTemp(i, :)), size(cLRTemp(i, :))))/sigComp^2);
        cLRTemp(i, cLRSortInd) =0; %0.001*min(cLRTemp(i, :));
    end

    if(mod(i, N_Nodes) == 0)
        [cRLTempSort, cRLSortInd] = sort(cRLTemp(i, :), 'ascend');
        cRLSortInd(1:kThresh) = [];
        minCRLTemp = cRLTempSort(1);
        indMin = find(cRLTempSort == minCRLTemp);
        cRLTempSort(indMin) = [];
        sigComp = sigFac*sqrt(cRLTempSort(1) - minCRLTemp);                 
        cRLTemp(i, :) = exp(-1*(cRLTemp(i, :) - repmat(min(cRLTemp(i, :)), size(cRLTemp(i, :))))/sigComp^2);
        cRLTemp(i, cRLSortInd) = 0; %0.001*min(cRLTemp(i, :));
    else
        [cRLTempSort, cRLSortInd] = sort(cRLTemp(i, :), 'ascend');
        cRLSortInd(1:kThresh) = [];
        minCRLTemp = cRLTempSort(1);
        indMin = find(cRLTempSort == minCRLTemp);
        cRLTempSort(indMin) = [];
        sigComp = sqrt(cRLTempSort(1) - minCRLTemp);                 
        cRLTemp(i, :) = exp(-1*(cRLTemp(i, :) - repmat(min(cRLTemp(i, :)), size(cRLTemp(i, :))))/sigComp^2);
        cRLTemp(i, cRLSortInd) =0; %0.001*min(cRLTemp(i, :));
    end

    if(ceil(i/N_Nodes) == M_Nodes)
        [cDUTempSort, cDUSortInd] = sort(cDUTemp(i, :), 'ascend');
        cDUSortInd(1:kThresh) = [];
        minCDUTemp = cDUTempSort(1);
        indMin = find(cDUTempSort == minCDUTemp);
        cDUTempSort(indMin) = [];
        sigComp = sigFac*sqrt(cDUTempSort(1) - minCDUTemp);  
        cDUTemp(i, :) = exp(-1*(cDUTemp(i, :) - repmat(min(cDUTemp(i, :)), size(cDUTemp(i, :))))/sigComp^2);
        cDUTemp(i, cDUSortInd) = 0; %0.001*min(cDUTemp(i, :));
    else
        [cDUTempSort, cDUSortInd] = sort(cDUTemp(i, :), 'ascend');
        cDUSortInd(1:kThresh) = [];
        minCDUTemp = cDUTempSort(1);
        indMin = find(cDUTempSort == minCDUTemp);
        cDUTempSort(indMin) = [];
        sigComp = sqrt(cDUTempSort(1) - minCDUTemp);  
        cDUTemp(i, :) = exp(-1*(cDUTemp(i, :) - repmat(min(cDUTemp(i, :)), size(cDUTemp(i, :))))/sigComp^2);
        cDUTemp(i, cDUSortInd) =0; % 0.001*min(cDUTemp(i, :));
    end

    if(ceil(i/N_Nodes) == 1)
        [cUDTempSort, cUDSortInd] = sort(cUDTemp(i, :), 'ascend');
        cUDSortInd(1:kThresh) = [];
        minCUDTemp = cUDTempSort(1);
        indMin = find(cUDTempSort == minCUDTemp);
        cUDTempSort(indMin) = [];
        sigComp = sigFac*sqrt(cUDTempSort(1) - minCUDTemp);  
        cUDTemp(i, :) = exp(-1*(cUDTemp(i, :) - repmat(min(cUDTemp(i, :)), size(cUDTemp(i, :))))/sigComp^2);
        cUDTemp(i, cUDSortInd) =  0; %0.001*min(cUDTemp(i, :));
    else
        [cUDTempSort, cUDSortInd] = sort(cUDTemp(i, :), 'ascend');
        cUDSortInd(1:kThresh) = [];
        minCUDTemp = cUDTempSort(1);
        indMin = find(cUDTempSort == minCUDTemp);
        cUDTempSort(indMin) = [];
        sigComp = sqrt(cUDTempSort(1) - minCUDTemp);  
        cUDTemp(i, :) = exp(-1*(cUDTemp(i, :) - repmat(min(cUDTemp(i, :)), size(cUDTemp(i, :))))/sigComp^2);
        cUDTemp(i, cUDSortInd) = 0; % 0.001*min(cUDTemp(i, :));

    end
end


cDU = (cDUTemp.*(ones(noPatches) - eye(noPatches)));
cLR = (cLRTemp.*(ones(noPatches) - eye(noPatches)));
cUD = (cUDTemp.*(ones(noPatches) - eye(noPatches)));
cRL = (cRLTemp.*(ones(noPatches) - eye(noPatches)));
% 
% for i = 1:noPatches
%     cDU(i, :) = cDU(i, :)/ sum(cDU(i, :));
%     cUD(i, :) = cUD(i, :)/ sum(cUD(i, :));
%     cLR(i, :) = cLR(i, :)/ sum(cLR(i, :));
%     cRL(i, :) = cRL(i, :)/ sum(cRL(i, :));
% end
