function [] = projectPlot2D(points, vector, st)
	pvector = [1; -vector(1) / vector(2)];
	A = [vector'; pvector'];
	for i=1:size(points,1)
		B = [ points(i,:) *  vector ;0];
		S = A\B;
		plot([points(i,1), S(1)],[points(i,2),S(2)],st);
	end
end
