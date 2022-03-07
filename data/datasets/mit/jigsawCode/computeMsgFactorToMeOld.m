function nodes = computeMsgFactorToMeOld(nodes, msg, i, kStates, sigExcl)
% Computes the message from the global factor to node i

msgThis = msg;
% Nulling the ith component of the 1-m.  See the note
msgThis(i, :) = ones(1, kStates);
msgTemp = prod(msgThis, 1);
nodes{i}.msgFactorToMe = msgTemp;

% msgTemp = log(prod(msgThis, 1));
% nodes{i}.msgFactorToMe = exp(msgTemp/(sigExcl^2));