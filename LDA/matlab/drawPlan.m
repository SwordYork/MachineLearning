function [] = drawPlan(project)
p = project';
A = p(:,1:2);
B = p(:,3);
normv = [A\B;-1]

mx = 20 * 1.2;
my = 20 * 1.2;
[xx,yy]=ndgrid(0:mx,0:my);
z = (-normv(1)*xx - normv(2)*yy)/normv(3);
surf(xx,yy,z)
colormap(winter)
shading interp
hold on;
end