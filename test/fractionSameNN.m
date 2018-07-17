function frac = fractionSameNN(data1, data2, k)
%Calculate number of the same neighbours for data 2 as for data 1.
    n = size(data1, 1);
    %Calculate distance matrices
    dat = sum(data1 .^ 2, 2);
    dist1 = bsxfun(@plus, dat, dat') - 2 * (data1 * data1');
    dat = sum(data2 .^ 2, 2);
    dist2 = bsxfun(@plus, dat, dat') - 2 * (data2 * data2');
    % Put infinit to main diagonal
    dist1 = dist1 + diag(Inf(1, n));
    dist2 = dist2 + diag(Inf(1, n));
    % Sort distance matrices to select 
    [~, ind1] = sort(dist1);
    [~, ind2] = sort(dist2);
    % Calculate number of the same neighbours
    frac = 0;
    for p = 1:n
        frac = frac + sum(ismember(ind1(1:k, p), ind2(1:k, p)));
    end
    frac = frac / (n * k);
end