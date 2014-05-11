function [ noise_data ] = gendata(length, k, b, mu, sigma)
	X = unifrnd(-100,100,[length,1]);
	Y = X * k + b + normrnd(mu,sigma,[length,1]) .* (abs(X)-100) * mu / 3 ;
	noise_data = [X,Y];
end
