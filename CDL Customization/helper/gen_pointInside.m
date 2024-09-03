% Generate points inside the quadrilateral
% Input: UE_grid_n == cell (e.g. Peel and William street -> 1x2 cell)
%        num_div_x == number of points across latitude  
%        num_div_y == number of points across longtitude
function points = gen_pointInside(UE_grid_n, num_div_x, num_div_y)
    A = UE_grid_n.vertice{1}.location;
    B = UE_grid_n.vertice{2}.location;
    C = UE_grid_n.vertice{3}.location;
    D = UE_grid_n.vertice{4}.location;

    % Generate points inside the quadrilateral
    points = zeros(num_div_x * num_div_y, 2);  % Preallocate for efficiency
    index = 1;

    for i = 0:num_div_x
        for j = 0:num_div_y
            % Interpolation along the x-direction
            P1_x = (1 - i/num_div_x) * A(1) + (i/num_div_x) * B(1);
            P1_y = (1 - i/num_div_x) * A(2) + (i/num_div_x) * B(2);

            P2_x = (1 - i/num_div_x) * D(1) + (i/num_div_x) * C(1);
            P2_y = (1 - i/num_div_x) * D(2) + (i/num_div_x) * C(2);

            % Interpolation along the y-direction
            point_x = (1 - j/num_div_y) * P1_x + (j/num_div_y) * P2_x;
            point_y = (1 - j/num_div_y) * P1_y + (j/num_div_y) * P2_y;

            points(index, :) = [point_x, point_y];
            index = index + 1;
        end
    end
    % % Display the points (optional)
    % scatter(points(:,1), points(:,2), '.');
    % axis equal;
    % 
    % figure;
    % geoplot(points(:,1), points(:,2), '.');
    % title('Vertices of the Street: William');
    % grid on;
end