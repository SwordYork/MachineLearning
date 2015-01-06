function [] = plotgauss(mu,Sigma,x,y)
	[X,Y] = meshgrid(x,y);
	r = [X(:) Y(:)];	
	z = zeros(length(r),1);
	for i = 1:length(r)
		z(i) = 1/(2/pi)/det(Sigma)*exp( -1/2*(r(i,:)-mu)*inv(Sigma)*(r(i,:)-mu)');
	end
	z = reshape(z,length(y),length(x));
	sf = surf(X,Y,z);
    shading interp;
    set(sf,'facealpha',0.5)
    xlabel('x');
    ylabel('y');
end
