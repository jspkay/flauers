%P = [1 0 0; 0 1 0;]
P = [-1 1 1;
    1 0 1]

pi_t = [1 1 1]

T = [P; pi_t]

N1 = 28
N2 = 28
N3 = 28

points = zeros(N1*N2*N3,2)

l = 1
for i = 1:N1
  for j = 1:N2
    for k = 1:N3
      z = P * [i, j, k]';
      x = z(1);
      y = z(2);
      points(l,:) = z';
      l = l+1;
      %printf("%3d %3d\n", x, y)
    endfor
  endfor
endfor

scatter(points(:,1), points(:,2), "*")
 
