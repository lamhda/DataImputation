% big test for database in matrix data
n = numel(data);

%  5% of gaps
% Produce gaps
gap = randsample(n, floor(n * 0.05));
gData = data;
gData(gap) = NaN;
clear gap
% Save incomplete data
save('gDataBlC05.mat', 'gData', '-v7.3');
% Calculate statistics
resBl05 = oneDBTest(gData, data);
% Save results
save('ResBlC05.mat', 'resBl05', '-v7.3');

% 10%
% Produce gaps
gap = randsample(n, floor(n * 0.1));
gData = data;
gData(gap) = NaN;
clear gap
% Save incomplete data
save('gDataBlC10.mat', 'gData', '-v7.3');
% Calculate statistics
resBl10 = oneDBTest(gData, data);
% Save results
save('ResBlC10.mat', 'resBl10', '-v7.3');

% 20%
% Produce gaps
gap = randsample(n, floor(n * 0.2));
gData = data;
gData(gap) = NaN;
clear gap
% Save incomplete data
save('gDataBlC20.mat', 'gData', '-v7.3');
% Calculate statistics
resBl20 = oneDBTest(gData, data);
% Save results
save('ResBlC20.mat', 'resBl20', '-v7.3');
