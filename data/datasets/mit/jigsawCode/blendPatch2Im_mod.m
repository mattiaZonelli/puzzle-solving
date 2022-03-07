function [imOut, gradMask, poissonMask] = blendPatch2Im_mod(patchSize, overlap, M_Nodes, N_Nodes, patch, nodeReconst, gradComp, lambda, compDU, compLR)

imOut = zeros(M_Nodes*(patchSize-overlap)+overlap, N_Nodes*(patchSize-overlap)+overlap, 3);
gradMask = zeros([size(imOut, 1), size(imOut, 2)]);
poissonMask = zeros([size(imOut, 1), size(imOut, 2)]);

for i = 1:M_Nodes
    for j = 1:N_Nodes

         M = ones(patchSize);
         MMaskGrad = ones(patchSize);
         PMaskGrad = zeros(patchSize);
         
         MGrad = zeros(patchSize);
         MyGrad = zeros(patchSize);
         MxGrad = zeros(patchSize);
         
         %We have a left overlap
         if( j > 1 )
%              if(nodeReconst(i, j) == nodeReconst(i, j-1) +1)
            if(compLR(nodeReconst(i, j), nodeReconst(i, j-1)) == 0)
                 Cy = ones(patchSize, overlap);
                 Cy(:, 1) = zeros(patchSize, 1);
                 fCorrect = 1;
             else
                 
                 E = sum((imOut(1+(i-1)*(patchSize-overlap) :(i-1)*(patchSize-overlap) + patchSize, ...
                     1+(j-2)*(patchSize-overlap) + patchSize - overlap:(j-2)*(patchSize-overlap) + patchSize, :)  - patch(:, 1:overlap, :, nodeReconst(i,j))).^2, 3);

                 %Compute the mincut array
                 [Cy] = mincut(E, 0);
                 fCorrect = 0;
                 PMaskGrad(1:end, 1:overlap) = double(Cy == 0);  
             end

             %Compute the mask and write to the destination
             M(1:end, 1:overlap) = double(Cy > 0);
             MMaskGrad(1:end, 1:overlap) = double(Cy >= 0);             
             MGrad(1:end, 1:overlap) = double(Cy == 0);   
         end;

         %We have a top overlap
         if( i > 1 )

%              if(nodeReconst(i, j) == nodeReconst(i-1, j) +N_Nodes)
             if(compDU(nodeReconst(i-1, j), nodeReconst(i, j)) == 0)
                 Cx = ones(overlap, patchSize);
                 Cx(1, :) = zeros(1, patchSize);
                 fCorrect = 1;
             else
                 E = sum(( imOut(1+(i-2)*(patchSize-overlap) + patchSize - overlap:(i-2)*(patchSize-overlap) + patchSize, 1+(j-1)*(patchSize-overlap):(j-1)*(patchSize-overlap) + patchSize, :) ...
                     - patch(1:overlap,:, :, nodeReconst(i, j))).^2, 3);

                 %Compute the SSD in the border region

                 %Compute the mincut array
                 Cx = mincut(E, 1);
                 fCorrect = 0;
 
%                  PMaskGrad(1:overlap, 1:end) =  MMaskGrad(1:overlap, 1:end).*(PMaskGrad(1:overlap, 1:end)) + MMaskGrad(1:overlap, 1:end) .* double(Cx == 0);             
             end

             %Compute the mask and write to the destination
             M(1:overlap, 1:end) = M(1:overlap, 1:end) .* double(Cx >0);
             MMaskGrad(1:overlap, 1:end) = MMaskGrad(1:overlap, 1:end) .* double(Cx >=0);             
             MGrad(1:overlap, 1:end) = (MMaskGrad(1:overlap, 1:end)) .* MGrad(1:overlap, 1:end)  + MMaskGrad(1:overlap, 1:end) .* double(Cx == 0);             
            if(fCorrect == 0)
                PMaskGrad(1:overlap, 1:end) =  MMaskGrad(1:overlap, 1:end).*(PMaskGrad(1:overlap, 1:end)) + MMaskGrad(1:overlap, 1:end) .* double(Cx == 0);
            end
             
             %              MxGrad(1:overlap, 1:end) = double(Cx == 0);
             
             
         end;


         if( i == 1 && j == 1 )
             imOut(1:patchSize, 1:patchSize, :) = patch(:, :, :, nodeReconst(1, 1));
             gradMask(1:patchSize, 1:patchSize) = MGrad;
             poissonMask(1:patchSize, 1:patchSize) = MGrad;
         else
             
             
             %Write to the destination using the mask
             imOut(1+(i-1)*(patchSize-overlap) :(i-1)*(patchSize-overlap) + patchSize, 1+(j-1)*(patchSize-overlap):(j-1)*(patchSize-overlap) + patchSize, :) = ...
                 filtered_write(imOut(1+(i-1)*(patchSize-overlap) :(i-1)*(patchSize-overlap) + patchSize, 1+(j-1)*(patchSize-overlap):(j-1)*(patchSize-overlap) + patchSize, :), ...
                 patch(:, :, :, nodeReconst(i, j)), M); 
             
             gradMask(1+(i-1)*(patchSize-overlap) :(i-1)*(patchSize-overlap) + patchSize, 1+(j-1)*(patchSize-overlap):(j-1)*(patchSize-overlap) + patchSize) = ...
                         filtered_write(gradMask(1+(i-1)*(patchSize-overlap) :(i-1)*(patchSize-overlap) + patchSize, 1+(j-1)*(patchSize-overlap):(j-1)*(patchSize-overlap) + patchSize), ...
                         MGrad, MMaskGrad);

            poissonMask(1+(i-1)*(patchSize-overlap) :(i-1)*(patchSize-overlap) + patchSize, 1+(j-1)*(patchSize-overlap):(j-1)*(patchSize-overlap) + patchSize) = ...
                     filtered_write(poissonMask(1+(i-1)*(patchSize-overlap) :(i-1)*(patchSize-overlap) + patchSize, 1+(j-1)*(patchSize-overlap):(j-1)*(patchSize-overlap) + patchSize), ...
                     PMaskGrad, PMaskGrad);

         end;              
    end 
end