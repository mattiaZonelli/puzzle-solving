function [cDU, cLR, cUD, cRL] = compComputeNO(M_Nodes, N_Nodes, compDU, compLR, DUClrDist, LRClrDist, logComp, ...
    noPatches, sigColor, directed, kThresh, patchRemove, patchAdd)

nNodes = N_Nodes * M_Nodes;
warning off all;

sigFac = 1;

%%
for k = 1:length(patchAdd)
    compLR(:, end+1) = compLR(:, patchAdd(k));
    compLR(end+1,:) = compLR(patchAdd(k), :);
    compDU(:, end+1) = compDU(:, patchAdd(k));
    compDU(end+1,:) = compDU(patchAdd(k), :);
end

% Removing the patches we wanted to get rid of
compDU(patchRemove, :) = [];
compDU(:, patchRemove) = [];
compLR(patchRemove, :) = [];
compLR(:, patchRemove) = [];

cDUTemp = compDU;
cUDTemp = compDU';
cLRTemp = compLR;
cRLTemp = compLR';

cLR = zeros(size(compLR));
cRL = zeros(size(compLR));
cDU = zeros(size(compDU));
cUD = zeros(size(compDU));

compThresh = 1;
% sigFac = 2;

if(logComp == 0)
    %normalizing Compatibility Matrices
    for i = 1:noPatches
        cLR(i, :) = (cLRTemp(i, :))/sum(cLRTemp(i, :));
        cRL(i, :) = (cRLTemp(i, :))/sum(cRLTemp(i, :));
        cDU(i, :) = (cDUTemp(i, :))/sum(cDUTemp(i, :));
        cUD(i, :) = (cUDTemp(i, :))/sum(cUDTemp(i, :));
    end
else
    if(directed)
%% Directed graph
        if(compThresh)
            
            cLRTemp = ones(size(compLR));
            cRLTemp = ones(size(compLR));
            cDUTemp = ones(size(compDU));
            cUDTemp = ones(size(compDU));

            %% Color information
            cDUColor = exp(-1*(DUClrDist-min(DUClrDist(:)))/(sigColor.^2));
            cLRColor = exp(-1*(LRClrDist - min(LRClrDist(:)))/(sigColor.^2));
            cUDColor = exp(-1*DUClrDist'- min(DUClrDist(:))/(sigColor.^2));
            cRLColor = exp(-1*LRClrDist'- min(LRClrDist(:))/(sigColor.^2));
            
            
            if kThresh < 25
                cDU = sparse(cDUTemp);%.*(ones(noPatches) - eye(noPatches)));
                cLR = sparse(cLRTemp);%.*(ones(noPatches) - eye(noPatches)));
                cUD = sparse(cUDTemp);%.*(ones(noPatches) - eye(noPatches)));
                cRL = sparse(cRLTemp);%.*(ones(noPatches) - eye(noPatches)));
            else
                cDU = (cDUTemp.*(ones(noPatches) - eye(noPatches))).*cDUColor;
                cLR = (cLRTemp.*(ones(noPatches) - eye(noPatches))).*cLRColor;
                cUD = (cUDTemp.*(ones(noPatches) - eye(noPatches))).*cUDColor;
                cRL = (cRLTemp.*(ones(noPatches) - eye(noPatches))).*cRLColor;
            end
            
            for i = 1:noPatches
                if(sum(cDU(i, :))~=0)
                    cDU(i, :) = cDU(i, :)/ sum(cDU(i, :));
                else
                    cDU(i, :) = ones(1, noPatches)/noPatches;
                end

                if(sum(cUD(i, :))~=0)
                    cUD(i, :) = cUD(i, :)/ sum(cUD(i, :));
                else
                    cUD(i, :) = ones(1, noPatches)/noPatches;
                end

                if(sum(cLR(i, :))~=0)
                    cLR(i, :) = cLR(i, :)/ sum(cLR(i, :));
                else
                    cLR(i, :) = ones(1, noPatches)/noPatches;
                end

                if(sum(cRL(i, :))~=0)
                    cRL(i, :) = cRL(i, :)/ sum(cRL(i, :));
                else
                    cRL(i, :) = ones(1, noPatches)/noPatches;
                end
                                
                
            end

        else
%             
% 
%             for i = 1:noPatches
%                 cLRTemp(i, :) = exp(-1*(cLRTemp(i, :) - repmat(min(cLRTemp(i, :)), size(cLRTemp(i, :))))/sigComp^2);
% 
%                 cRLTemp(i, :) = exp(-1*(cRLTemp(i, :) - repmat(min(cRLTemp(i, :)), size(cRLTemp(i, :))))/sigComp^2);
% 
%                 cDUTemp(i, :) = exp(-1*(cDUTemp(i, :) - repmat(min(cDUTemp(i, :)), size(cDUTemp(i, :))))/sigComp^2);
% 
%                 cUDTemp(i, :) = exp(-1*(cUDTemp(i, :) - repmat(min(cUDTemp(i, :)), size(cUDTemp(i, :))))/sigComp^2);
% 
%             end
% 
%             cDU =  sparse(cDUTemp.*(ones(noPatches) - eye(noPatches)));
%             cLR =  sparse(cLRTemp.*(ones(noPatches) - eye(noPatches)));
%             cUD =  sparse(cUDTemp.*(ones(noPatches) - eye(noPatches)));
%             cRL =  sparse(cRLTemp.*(ones(noPatches) - eye(noPatches)));
% 
%             for i = 1:noPatches
%                 cDU(i, :) = cDU(i, :)/ sum(cDU(i, :));
%                 cUD(i, :) = cUD(i, :)/ sum(cUD(i, :));
%                 cLR(i, :) = cLR(i, :)/ sum(cLR(i, :));
%                 cRL(i, :) = cRL(i, :)/ sum(cRL(i, :));
%             end
        end
    else
% 
% 
% %% Undirected graph
% %     
%         cLRTemp = cLRTemp - repmat(min(cLRTemp(:)), size(cLRTemp));
%         cRLTemp = cRLTemp - repmat(min(cRLTemp(:)), size(cRLTemp));
%         cDUTemp = cDUTemp - repmat(min(cDUTemp(:)), size(cDUTemp));
%         cUDTemp = cUDTemp - repmat(min(cUDTemp(:)), size(cUDTemp));
% 
% 
%         cDU = exp(-1*(cDUTemp.*(ones(noPatches)))/sigComp^2);
%         cLR = exp(-1*(cLRTemp.*(ones(noPatches)))/sigComp^2);
%         cUD = exp(-1*(cUDTemp.*(ones(noPatches)))/sigComp^2);
%         cRL = exp(-1*(cRLTemp.*(ones(noPatches)))/sigComp^2);
% 
% 
%         cDU = cDU.*(ones(noPatches) - eye(noPatches));
%         cLR = cLR.*(ones(noPatches) - eye(noPatches));
%         cUD = cUD.*(ones(noPatches) - eye(noPatches));
%         cRL = cRL.*(ones(noPatches) - eye(noPatches));
    end
end
