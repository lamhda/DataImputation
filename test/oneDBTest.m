function res = oneDBTest(data, orig)
%oneDBTest perform test of three methods for one database and report
%results.
%
%Inputs:
%   data is n-by-m matrix of data with gaps.
%   orig is n-by-m matrix of original complete data
%
%Outputs:
%   res is 3-by-k matrix of results. Each row corresponds to one method:
%           1 10NN
%           2 Unrestricted SVD
%           3 Restricted SVD
%       Each column corresponds to one of metrics
%           1  time spent
%           2  10FSNN d d
%           3  20FSNN d d
%           4  10FSNN 3 3
%           5  20FSNN 3 3
%           6  10FSNN d 3
%           7  20FSNN d 3
%           8  Angle
%           9  Length
%           10 ttest data
%           11 ftest data
%           12 ttest 3PC
%           13 ftest 3PC

    % Create output matrix
    res = zeros(3, 13);

    % Calculate parameters for original database
    [VGT, data3PC] = pc3(orig);
    
    % Perform 10NN
    fprintf('10NN\n');
    tic;
    knnData = kNNImpute(data, 10);
    res(1, 1) = toc;
    
    % Calculate projections
    [VkNN, knn3PC] = pc3(knnData);
    
    % Calculate statistics
    res(1, 2:end) = calcStat(orig, VGT, data3PC, knnData, VkNN, knn3PC);
    
    %Perform USVD
    fprintf('USVD\n');
    tic;
    SVDataI = svdWithGaps(data, 'interval', 'infinit');
    res(2, 1) = toc;
    
    % Calculate projections
    [VSVDI, svdI3PC] = pc3(SVDataI);
    
    % Calculate statistics
    res(2, 2:end) = calcStat(orig, VGT, data3PC, SVDataI, VSVDI, svdI3PC);

    %Perform RSVD
    fprintf('RSVD\n');
    tic;
    SVDataR = svdWithGaps(data);
    res(3, 1) = toc;
    
    % Calculate projections
    [VSVDR, svdR3PC] = pc3(SVDataR);
    
    % Calculate statistics
    res(3, 2:end) = calcStat(orig, VGT, data3PC, SVDataR, VSVDR, svdR3PC);
end

function [V, dat] = pc3(data)
    [~, D, V] = svds(data, 3);
    D = diag(D);
    [~, ind] = sort(D, 'descend');
    V = V(:, ind);
    dat = data * V;
end

function res = calcStat(dat1, V1, dat3_1, dat2, V2, dat3_2)
%           1  10FSNN d d
%           2  20FSNN d d
%           3  10FSNN 3 3
%           4  20FSNN 3 3
%           5  10FSNN d 3
%           6  20FSNN d 3
%           7  Angle
%           8  Length
%           9  ttest data
%           10 ftest data
%           11 ttest 3PC
%           12 ftest 3PC
    res = zeros(1, 12);
    res(1) = fractionSameNN(dat1, dat2, 10);
    res(2) = fractionSameNN(dat1, dat2, 20);
    res(3) = fractionSameNN(dat3_1, dat3_2, 10);
    res(4) = fractionSameNN(dat3_1, dat3_2, 20);
    res(5) = fractionSameNN(dat1, dat3_2, 10);
    res(6) = fractionSameNN(dat1, dat3_2, 20);
    mat = abs(V1' * V2);
    res(7) = sum(acos(max(mat)));
    res(8) = sum(sqrt(sum(mat .^ 2)));
    [~, tt] = ttest2(dat1, dat2);
    res(9) = sum(tt < 0.05) / numel(tt);
    [~, tt] = vartest2(dat1, dat2);
    res(10) = sum(tt < 0.05) / numel(tt);
    [~, tt] = ttest2(dat3_1, dat3_2);
    res(11) = sum(tt < 0.05) / numel(tt);
    [~, tt] = vartest2(dat3_1, dat3_2);
    res(12) = sum(tt < 0.05) / numel(tt);
end