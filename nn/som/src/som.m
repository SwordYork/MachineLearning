f = fopen('hw4-data.txt');
data = textscan(f, '%f %f');
Vdata = [data{1}';data{2}'];

n = 5;
l = 1;
eps0 = 4;
lambda = 1000;

W = rand(2,n^2);

iter_limit = 4000;

%% -----------------------------------------------------
% SOM main loop
for current_iter=1:iter_limit
    input = randi(size(Vdata,2), 1);

    Wt = sum((W - repmat(Vdata(:,input), 1, n^2)).^2);
    [~, wins] = min(Wt);
    
    for i=1:size(W,2)
        dist = norm(mod(wins-1, n) - mod(i-1, n), 1) + norm(floor((wins-1)/n) - floor((i-1)/n), 1);
        rel_update = exp(-dist^2 / 2 / (eps0 * exp(-current_iter/lambda)));
        W(:,i) = W(:,i) + rel_update * l * exp(-current_iter/lambda)*(Vdata(:,input) - W(:,i));
    end
    
    if rel_update == 0
        break
    end
end

current_iter
rel_update

%% -----------------------------------------
% plot result

plot(Vdata(1,:),Vdata(2,:),'.');
hold on;

% 
for i=1:n
    plot(W(1, 1+(i-1)*n:i*n ),W(2,1+(i-1)*n:i*n),'ro-');
end

for i=1:n
    for j=0:n-2
        plot([W(1, i+j*n), W(1, i+(j+1)*n)],[W(2, i+j*n), W(2, i+(j+1)*n)],'ro-');
    end
end

hold off;


