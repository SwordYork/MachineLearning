points = 7;
OriginPoint = rand(points,2) * 10;
OriginDistance = caldist(OriginPoint);

subplot (1, 2, 1);
plotdist(OriginPoint,OriginDistance);
title ('Origin Points');

%%%%%%%%%%%%%%
% reverse

[V1,L1,X] = mds(OriginDistance);
MdsDistance = caldist(X);
subplot (1, 2, 2);
plotdist(X,MdsDistance);
title ('MDS Points');
