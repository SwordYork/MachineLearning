function [] = plotdiff(OriginM,NewM)
    s = size(OriginM)
    tx = 1:s(1);
    ty = 1:s(2);
    [xx, yy] = meshgrid (tx, ty); 
    stem3(xx',yy',OriginM,'bd-.','markersize', 10,'linewidth', 5)
    hold on;
    grid on;
    stem3(xx',yy',NewM,'g-.','markersize', 10,'linewidth', 6)
    axis([min(xx(:))-0.5,max(xx(:))+0.5,min(yy(:))-0.5,max(yy(:))+0.5])
    h1 = surf(xx',yy',xx'*0)
    set(h1,'facealpha',0.5);
    colormap(winter)
end