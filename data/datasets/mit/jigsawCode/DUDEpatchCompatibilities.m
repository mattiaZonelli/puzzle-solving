function D = DUDEpatchCompatibilities(patch,w)

patchSize = size(patch,2);
n = size(patch,4);
D = zeros(n,n);
L = zeros(n,n);
R = zeros(n,n);

for k=1:length(w),
    fprintf(1,'Iteration %d/%d\n',k,length(w));
    kk = w(k);
    for i=1:n,
        L(i,:) = partialPatchDist(patch(:,end+1-kk:end,:,i),patch(:,1:kk,:,:));
        R(i,:) = partialPatchDist(patch(:,1:patchSize-kk,:,i),patch(:,end+1-(patchSize-kk):end,:,:));
    end
   
    for i=1:n,
        for j=1:n,
            D(i,j) = D(i,j) + min(L(i,:)+R(j,:));
        end
%         D(i,i) = 0;
    end
end

return


function Z = MRFLineSegCrossover(Y,C,d2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Each column of Y is a line segment
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

d = size(Y,1);
n = size(Y,2);

% Preprocess
d1 = d-d2;
L = clusterDist(Y(d2+1:end,:),C(1:d1,:));
R = clusterDist(Y(1:d2,:),C(d1+1:end,:));

maxVal = max([L(:);R(:)])+1;
ONE = ones(1,n);
Z = zeros(d,n);
for i=1:n,
    D = L(:,i)*ONE + R;
    D(:,i) = maxVal;
    [dummy,j] = min(min(D,[],1));
    Z(:,i) = [Y(1:d2,i);Y(d2+1:end,j)];
    % Smooth out the border line
%    Z(d2:d2+1,i) = mean(Z(d2:d2+1,i));
end

return
