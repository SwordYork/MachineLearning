function [ D ]  = caldist(X)

  points = size(X)(1);
  for i = 1:points
    for j = 1:points
      D(i,j) = sum(  (X(i,:)-X(j,:)).^2 ) ^0.5;
    end
  end

end
