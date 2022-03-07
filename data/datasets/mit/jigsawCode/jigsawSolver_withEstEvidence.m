clear;
 

imNo = 1;
%% load image
imName = sprintf('%s%s', './imData/', num2str(imNo), '.png'); 
inputIm = im2double(imread(imName));

%% Image cutting, evidence and compatibility computation

patchSize = 28;
sz = size(inputIm);
M_patches = floor(sz(1)/patchSize);
N_patches = floor(sz(2)/patchSize);
noPatches =  M_patches*N_patches; 


% Image cutting, evidence and compatibility computation
% Cutting the image into patches

inputImGray = rgb2gray(inputIm);
inputImNTSCT = rgb2ntsc(inputIm);

% normalizing the NTSC channels to equalize the variance
inputImNTSC = inputImNTSCT(:, :, 1);
inputImNTSC(:, :, 2) = 7*inputImNTSCT(:, :, 2);
inputImNTSC(:, :, 3) = 7*inputImNTSCT(:, :, 3);

% Gridding into patches
patchGray = cutImintoPatch(patchSize, patchSize, N_patches, M_patches, inputImGray);
patchNTSC = cutImintoPatchRGB(patchSize, patchSize, N_patches, M_patches, inputImNTSC);
patch = cutImintoPatchRGB(patchSize, patchSize, N_patches, M_patches, inputIm);

% image resizing for evidence/cluster estimation
inputImS = imresize(inputIm, 7/28);
patchDown = cutImintoPatchRGBOverlap(7, N_patches, M_patches, 0, inputImS);

% computing the naive energy for the compatibility
tic;
    [x, y, DUClrDist, LRClrDist] = compCompatibilityColor(patchNTSC, patchSize, noPatches);
toc;

compDU = DUClrDist;
compLR = LRClrDist;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Computing the evidence
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Loading the global clusters
clusterFileName = sprintf('%s%s%s%s%s', './data/globalClusterColorFastSub_200.mat');
load('-MAT', clusterFileName);

% Loading the PCA components
eigenCompFileName = sprintf('%s%s%s%s%s', './data/PCA_ColorFast.mat');
load(eigenCompFileName);

%Loading the total number of images
noImagesClusterName = sprintf('%s%s%s%s%s', './data/noImages.mat');
load(noImagesClusterName);

% Specifying the number of clusters
noCluster =200;

% Projecting all patches onto the PCA basis
imDatasetTest = patchRas(patchDown);
imDatasetTestPCA = EigenComp'*(imDatasetTest - repmat(meanVec, 1, size(imDatasetTest, 2)));

% taking the top 22 coefficients (98.5% variance)
imDatasetTestPCAT = imDatasetTestPCA(1:22, :);

% Computing the distance between the patch representation to the
% cluster centers
distanceVec = zeros(noPatches, noCluster);
for i = 1:noCluster
    distanceVec(:, i) = calcdist(imDatasetTestPCAT',centers(i, :));
end

% assign each patch to one of the 200 clusters
[minDistanceTest, minDistInd]= min(distanceVec, [], 2);
patchLabel = reshape(minDistInd, [N_patches, M_patches])';

clusterLabelOrig = minDistInd;

%% Estimating the evidence image

% Generate the histogram representation of the image
histIm = hist(patchLabel(:), noCluster);

% lo
ARegressFile = sprintf('%s%s%s%s%s','./data/ARegressColorFastSub_200.mat');
load(ARegressFile);

% Run regression to estimate the scene
YA = A*histIm(:);
evidence = reshape(YA, [M_patches, N_patches, 3]);

figure, imshow(evidence)



M_Nodes = M_patches;
N_Nodes = N_patches;

% loading the prob map
clusterFileName = sprintf('%s%s%s%s%s', './data/probMapColorFast_200.mat');
load(clusterFileName);
probMap = ones(M_Nodes, N_Nodes, noCluster);
for clusterInd = 1:noCluster
    % Probability map
    probMapT = clusterMask(:, :,(clusterInd));
    probMap(:, :, clusterInd) = probMapT/sum(probMapT(:));
%         probMap(:, :, clusterInd) = probMapT;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% running BP to reconstruct the image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M_Nodes = M_patches; % the number of vertical nodes
N_Nodes = N_patches; % the number of horizontal nodes
kStates = noPatches;
exclusive = 1;
patchDet = [];
sigEvid = 0.4;       % the sigma value for local evidence
maxProductFlag =0;

withEvidence = 1;    % with local evidence
likeClr = [];
sigClr = 1;
likeClrOn = 0;
blend = 1;
alpha = 0.5;
sigExcl = 1;
overlap = 0;

patchAdd =[];
patchRemove = [];
patchMoveTo = [];
patchMoveFrom = [];
directed = 1;
kThresh = noPatches;
sigColor = 0.25;
logComp = 1;

corrPatchTemp = reshape([1:noPatches], [N_patches, M_patches])';

noBPIter = 3;           % This number specifies, at the currently fixed patch configuration, how many times we want to run BP to reconstruct an image.
noFixStep = [0:2:10];   % This number specifies how many patches to fix to correct locations when we reconstruct the image
nIter = 100;            % The number of BP iterations
randInit = 1;           % Random BP message initialization
pairCorr = zeros(length(noFixStep),noBPIter);   % For measuring the correct image reconstruction rate according to pair-wise correct measure
clusterCorr = zeros(length(noFixStep), noBPIter);% For measuring the correct image reconstruction rate according to cluster-wise correct measure
neighCorr = zeros(length(noFixStep), noBPIter);  % For measuring the correct image reconstruction rate according to neighbor-wise correct measure

% Loop over the number of patches to fix
for noFixInd = 1:length(noFixStep)
    
    % for different number of anchor patches, we choose different anchor
    % patch locations
    if(noFixInd == 1)
        patchFix = [];
        patchFixOrig = [];                
    elseif(noFixInd == 2)
        patchFix = [222, 234];
        patchFixOrig = patchFix;
    elseif(noFixInd == 3)
        patchFix = [126, 138, 342, 354];
        patchFixOrig = patchFix;
    elseif(noFixInd == 4)
        patchFix = [124, 132, 140, 340, 348, 356];
        patchFixOrig = patchFix;
    elseif(noFixInd == 5)
        patchFix = [76, 84, 92, 222, 234, 364, 372, 380];
        patchFixOrig = patchFix;
    elseif(noFixInd == 6)
        patchFix = [76, 84, 92, 364, 372, 380, 219, 225, 231, 237];
        patchFixOrig = patchFix;
    end

    % For coloring the anchor patches
    patchSig = patch;
    sigP = zeros(patchSize, patchSize, 3);
    sigP(:, :, 1) = ones(patchSize);

    for sigL = 1:length(patchFix)
        patchSig(:, :, :, patchFix(sigL)) = 0.5*patch(:, :, :, patchFix(sigL)) + 0.5*sigP;
    end
    
    
    % Looping over differnet BP runs
    for BPIter = 1:noBPIter

        tic;
        % MRF Initialization + compatibility initialization
        % nodes are enumerated in a horizontal-first scan fashion.
        K_states = noPatches;
        [nodes, cDU, cLR] = initMRFPatchWConst(M_Nodes, N_Nodes, K_states, compDU, compLR, 1, noPatches, 1, 1, noPatches, [], []);

        [nodes, evidMatrix, energyMat] = initPatchToNodeHis(nodes, patch, evidence, probMap, patchLabel, noPatches, sigEvid, patchFix, patchMoveFrom, patchMoveTo);

        % MRF node message initialization
        [nodes] = initBPMessagesWConst(nodes, K_states, exclusive, randInit);

        % message passing
        for i =1:nIter
            i
            nodes = oneIterBPDampWConstMod(nodes, maxProductFlag, alpha, K_states, exclusive, sigExcl);
        end

        nodes = computeBeliefsWConst(nodes,  1);
        [nodeReconst] = marginals2image(nodes, M_Nodes, N_Nodes);  %, noPatches - noRemove + noAdd)
        imOutTemp = blendPatch2Im(patchSize, patchSize, M_Nodes, N_Nodes, patchSig, nodeReconst);
        figure, imshow(imOutTemp)

        %% Compute scores

        [pairCorrOut, clusterCorrOut, neighCorrOut] = scoreCompute(nodeReconst, noPatches, M_patches, N_patches, patchLabel, corrPatchTemp, clusterLabelOrig);
        pairCorr(noFixInd, BPIter) = pairCorrOut;
        clusterCorr(noFixInd, BPIter) = clusterCorrOut;
        neighCorr(noFixInd, BPIter) = neighCorrOut;

    end
end

imNo
pairCorr
neighCorr
clusterCorr
