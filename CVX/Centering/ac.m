a1 = 1;
b1 = 1;

a2 = -3;
b2 = -3;

a3 = 0.2;
b3 = -2;

a4 = -2;
b4 = 4;


x = -3:0.1:3;
figure;
subplot(2,1,1);
plot(x, x*a1+b1, 'r');
hold on
plot(x, x*a2+b2, 'g');
plot(x, x*a3+b3, 'b');
plot(x, x*a4+b4, 'y');
axis([-3,3,-3,3])
[X, Y] = meshgrid(x);

f = @(x,y) (-(log(b1-a1*x-y) + log(b2-a2*x-y) + log(b3-a3*x-y) + log(b4-a4*x-y) ))

Z = real(f(X,Y));

indices = find(abs(Z)>10);
Z(indices) = 0;  
subplot(2,1,2)
surf(X,Y,Z);