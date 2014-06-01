function [] = plotguass(Sigma,x,y)
	[X,Y] = meshgrid(x,y);
	r = [X(:) Y(:)];	
	z = zeros(length(r),1);
	for i = 1:length(r)
		z(i) = 1/(2/pi)/det(Sigma)*exp( -1/2*r(i,:)*inv(Sigma)*r(i,:)');
	end
	z = reshape(z,length(y),length(x));
	mesh(X,Y,z);
end
