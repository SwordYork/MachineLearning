points = 9;
OriginPoint = rand(points,2) * 10;
OriginDistance = zeros(points,points);

for i = 1:points
  for j = 1:points
    OriginDistance(i,j) = sum(  (OriginPoint(i,:)-OriginPoint(j,:)).^2 ) ^0.5;
  end
end

subplot (1, 2, 1);
plotdist(OriginPoint,OriginDistance);
title ('Origin Points');

%%%%%%%%%%%%%%
% reverse

[V1,L1,X] = mds(OriginDistance);
MdsDistance = zeros(points,points);
for i = 1:points
  for j = 1:points
    MdsDistance(i,j) = sum(  (X(i,:)-X(j,:)).^2 ) ^0.5;
  end
end
subplot (1, 2, 2);
plotdist(X,MdsDistance);
title ('MDS Points');