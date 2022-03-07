function [cost] = seamEnergyCompute(X,dir)

if( nargin > 1 && dir == 1 )
    X = X';
end;

%Allocate the current cost array, and set first row to first row of X
E = zeros(size(X));
E(1:end,:) = X(1:end,:);

%Starting with the second array, compute the path costs until the end
for i=2:size(E,1),    
    E(i,1) = X(i,1) + min( E(i-1,1), E(i-1,2) );
    for j=2:size(E,2)-1,
        E(i,j) = X(i,j) + min( [E(i-1,j-1), E(i-1,j), E(i-1,j+1)] );
    end
    E(i,end) = X(i,end) + min( E(i-1,end-1), E(i-1,end) );    
end

[cost, idx] = min(E(end, 1:end));