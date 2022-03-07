function [pairCorrOut, clusterCorrOut, neighCorrOut] = scoreCompute(nodeReconst, noPatches, M_patches, N_patches, patchLabel, corrPatchTemp, clusterLabelOrig)



% Pairwise comparison with the ground truth
pairCorrOut = sum(nodeReconst(:) == corrPatchTemp(:))/noPatches;

% the average rate of correct neighbor classification
corrNeighMat = zeros(M_patches, N_patches);

for M = 1:M_patches
    for N = 1:N_patches
        pNo = nodeReconst(M, N);
        count = 0;

        if(M == 1)
            if(N == 1)

                if(nodeReconst(M, N+1) == pNo+1)
                    count = count+1;
                end

                if(nodeReconst(M+1, N) == pNo+N_patches)
                    count = count+1;
                end                               

                corrNeighMat(M, N) = count/2;

            elseif(N == N_patches)
                if(nodeReconst(M, N-1) == pNo-1)
                    count = count+1;
                end

                if(nodeReconst(M+1, N) == pNo+N_patches)
                    count = count+1;
                end                               

                corrNeighMat(M, N) = count/2;
            else
                if(nodeReconst(M, N-1) == pNo-1)
                    count = count+1;
                end

                if(nodeReconst(M, N+1) == pNo+1)
                    count = count+1;
                end

                if(nodeReconst(M+1, N) == pNo+N_patches)
                    count = count+1;
                end                               

                corrNeighMat(M, N) = count/3;                                
            end
        elseif(M==M_patches)
            if(N == 1)

                if(nodeReconst(M, N+1) == pNo+1)
                    count = count+1;
                end

                if(nodeReconst(M-1, N) == pNo-N_patches)
                    count = count+1;
                end                               

                corrNeighMat(M, N) = count/2;

            elseif(N == N_patches)
                if(nodeReconst(M, N-1) == pNo-1)
                    count = count+1;
                end

                if(nodeReconst(M-1, N) == pNo-N_patches)
                    count = count+1;
                end    

                corrNeighMat(M, N) = count/2;
            else
                if(nodeReconst(M, N-1) == pNo-1)
                    count = count+1;
                end

                if(nodeReconst(M, N+1) == pNo+1)
                    count = count+1;
                end

                if(nodeReconst(M-1, N) == pNo-N_patches)
                    count = count+1;
                end   

                corrNeighMat(M, N) = count/3;                                
            end
        else
            if(N==1)
                if(nodeReconst(M, N+1) == pNo+1)
                    count = count+1;
                end

                if(nodeReconst(M+1, N) == pNo+N_patches)
                    count = count+1;
                end

                if(nodeReconst(M-1, N) == pNo-N_patches)
                    count = count+1;
                end   

                corrNeighMat(M, N) = count/3;  

            elseif(N==N_patches)
                if(nodeReconst(M, N-1) == pNo-1)
                    count = count+1;
                end

                if(nodeReconst(M+1, N) == pNo+N_patches)
                    count = count+1;
                end

                if(nodeReconst(M-1, N) == pNo-N_patches)
                    count = count+1;
                end   

                corrNeighMat(M, N) = count/3;  

            else
                if(nodeReconst(M, N-1) == pNo-1)
                        count = count+1;
                end

                if(nodeReconst(M, N+1) == pNo+1)
                    count = count+1;
                end

                if(nodeReconst(M+1, N) == pNo+N_patches)
                    count = count+1;
                end   

                if(nodeReconst(M-1, N) == pNo-N_patches)
                    count = count+1;
                end   
                corrNeighMat(M, N) = count/4;   
            end

        end                      
    end
end

neighCorrOut = sum(corrNeighMat(:))/noPatches;

% Matching the cluster labels between the original label and the
% new label
clusterLabelReconst = clusterLabelOrig(nodeReconst);
clusterCorrOut = sum(clusterLabelReconst(:) == patchLabel(:))/noPatches;
