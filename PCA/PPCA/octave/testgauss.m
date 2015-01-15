x = linspace(-5,5,100);
y = linspace(-5,5,100);
u = [0,0];
sigma = [2,0;0,2];
subplot(2,3,1);
plotgauss(u,sigma,x,y);
title('u[0,0],sigma = [2,0;0,2]');

u = [2,3];
sigma = [2,0;0,2];
subplot(2,3,2);
plotgauss(u,sigma,x,y);
title('u[2,3],sigma = [2,0;0,2]');

u = [0,0];
sigma = [2,0;0,6];
subplot(2,3,3);
plotgauss(u,sigma,x,y);
title('u[0,0],sigma = [2,0;0,6]');

u = [0,0];
sigma = [2,1;1,2];
inv(sigma)
subplot(2,3,4);
plotgauss(u,sigma,x,y);
title('u[0,0],sigma = [2,1;1,2]');


u = [0,0];
sigma = [2,-1;-1,2];
inv(sigma)
subplot(2,3,5);
plotgauss(u,sigma,x,y);
title('u[0,0],sigma = [2,-1;-1,2]');

u = [0,0];
sigma = [2,1;-2,2];
inv(sigma)
subplot(2,3,6);
plotgauss(u,sigma,x,y);
title('u[0,0],sigma = [2,1;-2,2]');
