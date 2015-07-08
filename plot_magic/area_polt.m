x=0:0.01:2*pi;                  %#initialize x array
y1=sin(x);                      %#create first curve
y2=sin(x)+.5;                   %#create second curve
X=[x,fliplr(x)];                %#create continuous x value array for plotting
Y=[y1,fliplr(y2)];              %#create y values for out and then back
fill(X,Y,'b');                  %#plot filled area
