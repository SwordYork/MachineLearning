pkg load statistics;


D = [0 2^0.5 2^0.5 5^0.5; 2^0.5 0 2^0.5*2 1; 2^0.5 2*2^0.5 0 13^0.5;5^0.5 1 13^0.5 0];
n = size(D)(1);


printf('Build-in function octave:\n')
[Y,e] = cmdscale(D)




%%%%%%%
% my mds
%%%%%%%
printf('My function:\n--------------------------\n')
[V1,L1,X] = mds(D);
printf('cordinate:\n')
disp(X)


%%%%%%%%%%
%  recalculate distance
%%%%%%%%%%
CD = zeros(n,n);
for i=1:n
    for j=1:n
	CD(i,j) = sum(  (X(i,:)-X(j,:)).^2  );
    end
end


printf('D.^2:\n')
disp(D.^2)


printf('calculate distance:\n')
disp(CD)




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%      add new distance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

new_dis = transpose([3 5^0.5 17^0.5 2^0.5]);
new_D = [D new_dis];
new_D = [new_D;[transpose(new_dis) 0]];
n = size(new_D)(1);

%%%%%%%%%%%%%%
%  one way
%%%%%%%%%%%%%%
printf('Simple recalculate function:\n--------------------------\n')

%%%%%%%
% use the same algorithm to calculate new vector
% it is very expensive when origin distance matrix is large
% my recalculate mds
%%%%%%%
printf('Recalculate Add New Distance:\n--------------------------\n')
[V2,L2,X2] = mds(new_D);
printf('cordinate:\n')
disp(X2)


%%%%%%%%%%
%  recalculate distance
%%%%%%%%%%
CD = zeros(n,n);
for i=1:n
    for j=1:n
	CD(i,j) = sum(  (X2(i,:)-X2(j,:)).^2  );
    end
end


printf('D.^2:\n')
disp(new_D.^2)


printf('calculate distance:\n')
disp(CD)


%%%%%%%%%%%%%%%%%%%%%%%
% new way
%%%%%%%%%%%%%%%%%%%%%
H = eye(n-1) - ones(n-1,1)*ones(1,n-1)/(n-1);

  
alpha = sum(X.^2,2) - new_dis.^2;
new_v = transpose( L1^-1 * transpose(X) * H * alpha) / 2;
%new_V1 = [V1;new_v];
%new_X1 = new_V1*L1^-0.5
new_X1 = [X;new_v]
sum(new_X1)


%%%%%%%%%%
%  recalculate distance
%%%%%%%%%%
CD = zeros(n,n);
for i=1:n
    for j=1:n
	CD(i,j) = sum(  (new_X1(i,:)-new_X1(j,:)).^2  );
    end
end


printf('D.^2:\n')
disp(D.^2)


printf('calculate distance:\n')
disp(CD)


