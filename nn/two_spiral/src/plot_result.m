A = importdata('two_spiral_seq_result.txt');
num_hidden_uint = 12;
WV1 = zeros(3, num_hidden_uint);
WU1 = zeros(2, num_hidden_uint);
for j=1:num_hidden_uint
    for i=1:3
        WV1(i,j) = A(j*2-1,i);
       
    end
    for i=1:2
         WU1(i,j) = A(j*2, i);
    end
end

WV2 = zeros(num_hidden_uint+1, 1);
WU2 = zeros(num_hidden_uint, 1);

WV2(1) = A(num_hidden_uint * 2 + 1, 1);

for j=1:num_hidden_uint
   WV2(j+1) =  A((num_hidden_uint+j)*2, 1);
   WU2(j) =  A((num_hidden_uint+j)*2+1, 1);
end


[X, Y] = meshgrid(-3:0.1:3);
RX = reshape(X, numel(X), 1);
RY = reshape(Y, numel(Y), 1);
T = [RX RY];
TP = [ones(numel(RX),1) RX RY];

Z1 = 1 ./ (1 + exp(-(TP*WV1+T.^2*WU1)));
Z1P = [ones(numel(RX),1) Z1];
Z2 = 1 ./ (1 + exp(-(Z1P*WV2 + Z1.^2*WU2)));

Z = reshape(Z2, size(X));
surf(X, Y, Z)

