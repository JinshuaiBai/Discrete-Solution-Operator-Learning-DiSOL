function geom_datasets = generateData(params)
%GENERATEDATA Batch-generate geometry + boundary-condition input tensors (DiSOL).
%
% This function connects:
%   - create2DGeom(): random 2D closed geometry mask on an imgSize x imgSize grid
%   - selectBoundary(): extract the longest boundary and sample a contiguous segment
%
% For each geometry, it generates multiple boundary-condition masks by selecting
% different random contiguous boundary segments. The output is a 4D tensor:
%   geom_datasets: [N x 2 x H x W]
% where:
%   channel 1: geometry mask (logical; 1 = inside geometry)
%   channel 2: boundary mask  (uint8;   1 = selected boundary segment pixels)
%
% Required fields in params
%   params.ImgSize       : (int) square grid resolution H = W
%   params.ControlPoints : (int) number of random points used by create2DGeom()
%   params.alpha         : (double) shrink factor used by boundary(), 0 < alpha <= 1
%   params.NumDiffGeo    : (int) number of different geometries to generate
%
% Optional fields in params
%   params.num_change_boundary : (int) number of boundary-condition variants per geometry (default: 5)
%   params.length_limit        : (double) boundary segment fraction limit in (0,1], or >1 for full boundary (default: 0.25)
%   params.seed                : (int) RNG seed for reproducibility (default: [])
%   params.return_logical_geom : (bool) whether to keep geometry mask as logical (default: true)
%
% Notes
% - This function is designed for dataset generation without visualization.
%   If you need visualization, do it in a separate debug script to keep the
%   generation pipeline deterministic and side-effect free.
%

% ----------------------------
% Defaults
% ----------------------------
if ~isfield(params, 'num_change_boundary') || isempty(params.num_change_boundary)
    params.num_change_boundary = 5;
end
if ~isfield(params, 'length_limit') || isempty(params.length_limit)
    params.length_limit = 0.25;
end
if ~isfield(params, 'seed')
    params.seed = [];
end
if ~isfield(params, 'return_logical_geom') || isempty(params.return_logical_geom)
    params.return_logical_geom = true;
end

% ----------------------------
% Input validation
% ----------------------------
required = {'ImgSize','ControlPoints','alpha','NumDiffGeo'};
for i = 1:numel(required)
    if ~isfield(params, required{i})
        error('generateData:MissingParam', 'params.%s is required.', required{i});
    end
end

validateattributes(params.ImgSize, {'numeric'}, {'scalar','integer','>=',8});
validateattributes(params.ControlPoints, {'numeric'}, {'scalar','integer','>=',3});
validateattributes(params.alpha, {'numeric'}, {'scalar','>',0,'<=',1});
validateattributes(params.NumDiffGeo, {'numeric'}, {'scalar','integer','>=',1});
validateattributes(params.num_change_boundary, {'numeric'}, {'scalar','integer','>=',1});
validateattributes(params.length_limit, {'numeric'}, {'scalar','>',0});

% ----------------------------
% Reproducibility (optional)
% ----------------------------
if ~isempty(params.seed)
    rng(params.seed);
end

H = params.ImgSize;
W = params.ImgSize;
nGeom = params.NumDiffGeo;
nBd   = params.num_change_boundary;

% Total samples = geometries * boundary-variants
N = nGeom * nBd;

% Pre-allocate output tensor.
% Use uint8 for compactness and consistency with common ML pipelines.
geom_datasets = zeros(N, 2, H, W, 'uint8');

sample_idx = 0;

for geo_idx = 1:nGeom
    % ----------------------------
    % 1) Generate a random geometry mask
    % ----------------------------
    geo_img = create2DGeom(params);  % logical [H x W]

    geo_chan = logical(geo_img);

    % ----------------------------
    % 2) Generate multiple boundary-condition masks for this geometry
    % ----------------------------
    for b = 1:nBd
        sample_idx = sample_idx + 1;

        % Initialize boundary mask
        u_bd = zeros(H, W, 'logical');

        % Sample a contiguous boundary segment (row, col indices)
        [u_bd_idx, ~] = selectBoundary(geo_img, params.length_limit);

        % Rasterize boundary segment indices into a binary mask
        % u_bd_idx is uint8 [L x 2] with (row, col)
        for j = 1:size(u_bd_idx, 1)
            r = u_bd_idx(j, 1);
            c = u_bd_idx(j, 2);
            % Safety guard for indices (robustness for edge cases)
            if r >= 1 && r <= H && c >= 1 && c <= W
                u_bd(r, c) = logical(1);
            end
        end

        % Pack into output tensor: [N x 2 x H x W]
        geom_datasets(sample_idx, 1, :, :) = geo_chan;
        geom_datasets(sample_idx, 2, :, :) = u_bd;

        if params.draw_fig
            subplot(1,5,4)
            imshow(geo_chan)
            title('Geometry mask')
            subplot(1,5,5)
            imshow(u_bd)
            title('Selected boundary')
            set(gcf, 'Position', [100 100 1500 300]);
            pause(0.1)
        end
    end
end

end
