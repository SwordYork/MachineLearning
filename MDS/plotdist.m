function [] = plotdist(xy,D)
     n = size(xy)(1);
     A = ones(n,n);
     gplot(A,xy,'o-');
     for i = 1:n
          for j = i+1:n
              text( (xy(i,1)+xy(j,1))/2,(xy(i,2) + xy(j,2))/2, num2str( D(i,j))   )
          
          end
     end
end