function D = partialPatchDist(p,patch)

n = size(patch,4);
D = zeros(1,n);

p2 = sum(p(:).*p(:));
q  = reshape(patch,size(patch,1)*size(patch,2)*size(patch,3),size(patch,4));

D = p2 - 2*sum(repmat(p(:),[1,n]).*q) + sum(q.*q);

return