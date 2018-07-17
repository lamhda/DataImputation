% Test of fragments for optimisation
nans = isnan(gData);
ind = find(nans);
[n, m] = size(gData);
b = mean(gData, 'omitnan');
y = rand(1, m) - 0.5;
tic;
for p = 1:1006
    tmp = repmat(y, n, 1);
    tmp(nans) = NaN;
    x = sum(bsxfun(@minus, gData, b) .* tmp, 2, 'omitnan')...
        ./ sum(tmp .^ 2, 2, 'omitnan');
end
toc
ons = ones(1,n);
zData = gData;
zData(ind) = 0;
tic;
for p = 1:1006
    tmp1 = ons' * y;
    tmp1(ind) = 0;
    x1 = sum(bsxfun(@minus, zData, b) .* tmp, 2)...
        ./ sum(tmp .^ 2, 2);
end
toc
