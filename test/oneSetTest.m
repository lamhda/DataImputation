function res = oneSetTest(data, orig)
%oneDBTest perform test of three methods for one database and report
%results.
%
%Inputs:
%   data is n-by-m matrix of data with filled gaps.
%   orig is n-by-m matrix of original complete data
%
%Outputs:
%   res is 1-by-k matrix of results. 
%       Each column corresponds to one of metrics
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

    % Create output matrix
    res = zeros(3, 12);

    % Calculate parameters for original database
    [VGT, data3PC] = pc3(orig);
    % Calculate parameters for imputed database
    [V, PC3] = pc3(data);
    
    % Calculate statistics
    res = calcStat(orig, VGT, data3PC, data, V, PC3);
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