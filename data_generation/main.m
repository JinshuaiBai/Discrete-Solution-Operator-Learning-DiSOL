clc; clear; close all;

% =========================================================
% DiSOL Geometry + Boundary Dataset Generation (Main Script)
% =========================================================
% This script generates a batch of 2D random geometries and corresponding
% boundary-condition masks, following the procedure described in the SI.
%
% Output:
%   geom_datasets: [N x 2 x H x W] uint8 tensor
%     channel 1: geometry mask (1 = inside geometry)
%     channel 2: boundary mask  (1 = selected boundary pixels)
% =========================================================

%% -----------------------------
% Configuration
% ------------------------------
params.NumDiffGeo        = 10;     % number of different geometries
params.num_change_boundary = 5;    % number of boundary variants per geometry (used by generateData)
params.ImgSize           = 64;     % grid size (H=W)
params.ControlPoints     = 20;     % number of random points for boundary() control
params.alpha             = 0.8;    % shrink factor for boundary(), (0, 1]
params.length_limit      = 0.25;   % segment length limit (fraction); >1 means full boundary
params.draw_fig          = true;   % if true, visualize samples after generation

%% -----------------------------
% Generate dataset
% ------------------------------
geom_datasets = generateData(params);

%% -----------------------------
% Save dataset
% ------------------------------
out_dir = "./output_geom/";
if ~exist(out_dir, "dir")
    mkdir(out_dir);
end

out_file = fullfile(out_dir, "2DGeomData.mat");
save(out_file, "geom_datasets", "-v7.3");  % -v7.3 supports larger arrays

fprintf("Saved dataset to: %s\n", out_file);
fprintf("Dataset shape: [%d x %d x %d x %d]\n", ...
    size(geom_datasets,1), size(geom_datasets,2), size(geom_datasets,3), size(geom_datasets,4));

