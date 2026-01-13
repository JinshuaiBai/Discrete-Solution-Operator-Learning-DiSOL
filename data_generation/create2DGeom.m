function geo_img = create2DGeom(params)
%CREATE2DGEOM Generate a random 2D closed geometry mask on a Cartesian grid.
%
% This function reproduces the original geometry generation procedure:
%   1) Sample random scattered points in a centered coordinate system.
%   2) Extract a (possibly non-convex) boundary using boundary() with shrink factor params.alpha.
%   3) Fit a quadratic B-spline through the boundary control points (degree = 2).
%   4) Rasterize the resulting closed curve into a binary mask using inpolygon().
%
% Inputs
%   params.ControlPoints : number of random points used to define the shape (integer, >= 3)
%   params.ImgSize : grid resolution (params.ImgSize x params.ImgSize) (integer, >= 8)
%   params.alpha   : boundary shrink factor in boundary(), 0 < params.alpha <= 1
%
% Output
%   geo_img : logical matrix [params.ImgSize x params.ImgSize], true = inside geometry
%
% Notes
% - Coordinate system is centered at (0,0) with approximate range
%   [-(params.ImgSize-1)/2, +(params.ImgSize-1)/2] along each axis.
% - Requires Spline Toolbox for spmak/augknt/fnval.
%

% ----------------------------
% 1) Sample random points (centered coordinates)
% ----------------------------
x = (rand(1, params.ControlPoints) - 0.5) * (params.ImgSize - 1);  % x in [-(params.ImgSize-1)/2, +(params.ImgSize-1)/2]
y = (rand(1, params.ControlPoints) - 0.5) * (params.ImgSize - 1);  % y in [-(params.ImgSize-1)/2, +(params.ImgSize-1)/2]

% ----------------------------
% 2) Extract boundary indices using shrink factor params.alpha
% ----------------------------
k = boundary(x', y', params.alpha);

% Guard against rare degeneracy (too few boundary points).
% This keeps the same procedure but avoids spline construction failures.
if numel(k) < 3
    x = (rand(1, params.ControlPoints) - 0.5) * (params.ImgSize - 1);
    y = (rand(1, params.ControlPoints) - 0.5) * (params.ImgSize - 1);
    k = boundary(x', y', params.alpha);
end

% ----------------------------
% 3) Quadratic B-spline construction (fixed as in original code)
% ----------------------------
controlPoints = [x(k); y(k)];      % 2 x Nb control points
n = size(controlPoints, 2) - 1;    % number of control points minus 1
degree = 2;                        % fixed B-spline degree (original choice)
knots = augknt(1:n, degree + 1);   % knot vector

sp = spmak(knots, controlPoints);  % spline representation

% Sample spline curve densely to approximate boundary polygon
t = linspace(knots(1), knots(end), 1000);  % fixed sample count (original choice)
points = fnval(sp, t);                     % 2 x 1000 curve samples

% ----------------------------
% 4) Rasterize: test grid points inside the closed curve
% ----------------------------
halfSpan = (params.ImgSize - 1) / 2;
[xGrid, yGrid] = meshgrid( ...
    linspace(-halfSpan, halfSpan, params.ImgSize), ...
    linspace(-halfSpan, halfSpan, params.ImgSize) ...
    );

geo_img = inpolygon(xGrid, yGrid, points(1, :), points(2, :));
geo_img = logical(geo_img);

if params.draw_fig
    subplot(1,5,1)
    scatter(x,y,'filled'),axis equal,...
        axis(0.5*[-params.ImgSize params.ImgSize -params.ImgSize params.ImgSize]),box on
    title('Control points')
    subplot(1,5,2)
    plot(controlPoints(1,:), controlPoints(2,:), 'ro--', ...
        'LineWidth', 1.5, 'DisplayName', 'Control Points');
    axis equal,axis(0.5*[-params.ImgSize params.ImgSize -params.ImgSize params.ImgSize]),box on
    title('Selection control points')
    subplot(1,5,3)
    fnplt(sp, 'b', 2); 
    axis equal,axis(0.5*[-params.ImgSize params.ImgSize -params.ImgSize params.ImgSize]),box on
    title('Outline of geometry')
end

end
