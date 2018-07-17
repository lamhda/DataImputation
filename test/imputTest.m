% Tester of imputation methods
what = 3;
gaps = 0.05; %fraction of gaps
% 0 Load bladderCancer
% 1 Load breastCancer
% 2 Produce gaps
% 3 Projection of complete data to 3D
% 4 kNN repairing
% 5 Projection of kNN repairing data to 3D
% 6 Comparison kNN with original
% 7 ppca repairing
% 8 Calculate all PCs for original data
% 9 Calculate unrestricted SVD imputation
% 10 Comparison USVD with original
% 11 Calculate restricted SVD imputation
% 12 Comparison RSVD with original

% Take bladderCancer data for test
if what == 0
    load('C:\LocalData\em322\Apps\Matlab\Maps\bladderCancer.mat');
    data = d2fn';
    clear col d2fn;
end

% Take breastCancer data for test
if what == 1
    load('C:\LocalData\em322\Apps\Matlab\Maps\breastCancer.mat');
    data = d1n;
    clear col d1n;
end

% Produce gaps
if what == 2
    n = numel(data);
    gap = randsample(n, floor(n * gaps));
    gData = data;
    gData(gap) = NaN;
    nans = isnan(gData);
    gapped = sum(nans, 2) > 0;
    clear gap n
end

% Calculate projection of complete data onto 3D
if what == 3
    [~, D, VGT] = svds(data, 3);
    D = diag(D);
    [~, ind] = sort(D, 'descend');
    VGT = VGT(:, ind);
    data3PC = data * VGT;
    clear D ind
end

% 4 kNN repairing
if what == 4
    tic;
    [knnData, knnVar] = kNNImpute(gData, 10);
    toc
end

% 5 kNN repairing project onto 3 PCs
if what == 5
    [~, D, VkNN] = svds(knnData, 3);
    D = diag(D);
    [~, ind] = sort(D, 'descend');
    VkNN = VkNN(:, ind);
    knn3PC = knnData * VkNN;
    clear D ind
end

if what == 6
    % With original
    frO10 = fractionSameNN(data, knnData, 10);
    frO20 = fractionSameNN(data, knnData, 20);
    frPC10 = fractionSameNN(data3PC, knn3PC, 10);
    frPC20 = fractionSameNN(data3PC, knn3PC, 20);
    frOPC10 = fractionSameNN(data, knn3PC, 10);
    frOPC20 = fractionSameNN(data, knn3PC, 20);
    mat = VGT' * VkNN;
    ang = sum(acos(max(mat)));
    len = sum(sqrt(sum(mat .^ 2)));
    [~, ft] = vartest2(data, knnData);
    ft = sum(ft > 0.05) / numel(ft);
    [~, ftPC] = vartest2(data3PC, knn3PC);
    ftPC = sum(ftPC > 0.05) / numel(ftPC);
    [~, tt] = ttest2(data, knnData);
    tt = sum(tt > 0.05) / numel(tt);
    [~, ttPC] = ttest2(data3PC, knn3PC);
    ttPC = sum(ttPC > 0.05) / numel(ttPC);
    clear mat
end

if what == 7
    tic;
    [coeff, score, pcvar, mu] = ppca(gData, min(size(gData)));
%     [coeff, score, pcvar, mu] = ppca(data, min(size(data)));
    toc
    ppcaData = bsxfun(@plus, mu, score * coeff');
    clear coeff score pcvar mu
end

if what == 8
    tic;
%     [~, D, V] = svds(knnData, min(size(gData)));
    [~, D, V] = svd(data);
    toc
    clear D V
end

if what == 9
    tic;
    SVDataI = svdWithGaps(gData, 'interval', 'infinit', 'verbose', 1, 'tol', 0.5);
    toc
    [~, D, VSVDI] = svds(SVDataI, 3);
    D = diag(D);
    [~, ind] = sort(D, 'descend');
    VSVDI = VSVDI(:, ind);
    svdI3PC = SVDataI * VSVDI;
    clear D ind
end

if what == 10
    % With original
    frO10 = fractionSameNN(data, SVDataI, 10);
    frO20 = fractionSameNN(data, SVDataI, 20);
    frPC10 = fractionSameNN(data3PC, svdI3PC, 10);
    frPC20 = fractionSameNN(data3PC, svdI3PC, 20);
    frOPC10 = fractionSameNN(data, svdI3PC, 10);
    frOPC20 = fractionSameNN(data, svdI3PC, 20);
    mat = VGT' * VSVDI;
    ang = sum(acos(max(mat)));
    len = sum(sqrt(sum(mat .^ 2)));
    [~, ft] = vartest2(data, SVDataI);
    ft = sum(ft > 0.05) / numel(ft);
    [~, ftPC] = vartest2(data3PC, svdI3PC);
    ftPC = sum(ftPC > 0.05) / numel(ftPC);
    [~, tt] = ttest2(data, SVDataI);
    tt = sum(tt > 0.05) / numel(tt);
    [~, ttPC] = ttest2(data3PC, svdI3PC);
    ttPC = sum(ttPC > 0.05) / numel(ttPC);
    clear mat
end

if what == 11
    tic;
    SVDataR = svdWithGaps(gData, 'verbose', 1, 'tol', 0.5);
    toc
    [~, D, VSVDR] = svds(SVDataR, 3);
    D = diag(D);
    [~, ind] = sort(D, 'descend');
    VSVDR = VSVDR(:, ind);
    svdR3PC = SVDataR * VSVDR;
    clear D ind
end
if what == 12
    % With original
    frO10 = fractionSameNN(data, SVDataR, 10);
    frO20 = fractionSameNN(data, SVDataR, 20);
    frPC10 = fractionSameNN(data3PC, svdR3PC, 10);
    frPC20 = fractionSameNN(data3PC, svdR3PC, 20);
    frOPC10 = fractionSameNN(data, svdR3PC, 10);
    frOPC20 = fractionSameNN(data, svdR3PC, 20);
    mat = VGT' * VSVDR;
    ang = sum(acos(max(mat)));
    len = sum(sqrt(sum(mat .^ 2)));
    [~, ft] = vartest2(data, SVDataR);
    ft = sum(ft > 0.05) / numel(ft);
    [~, ftPC] = vartest2(data3PC, svdR3PC);
    ftPC = sum(ftPC > 0.05) / numel(ftPC);
    [~, tt] = ttest2(data, SVDataR);
    tt = sum(tt > 0.05) / numel(tt);
    [~, ttPC] = vartest2(data3PC, svdR3PC);
    ttPC = sum(ttPC > 0.05) / numel(ttPC);
    clear mat
end

