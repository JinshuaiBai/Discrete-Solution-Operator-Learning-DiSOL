function [selectedSegment, startIndex] = selectBoundary(bwImage, length_limit)
%SELECTBOUNDARY Extract a boundary from a binary image and sample a contiguous segment.
%
% This function:
%   1) Extracts object boundaries from a binary image using bwboundaries().
%   2) Selects the longest boundary (assumed to be the primary geometry).
%   3) Ensures the boundary is closed.
%   4) Randomly samples a contiguous boundary segment with wrap-around support.
%
% Inputs
%   bwImage      : binary image (logical or numeric). Nonzeros are treated as foreground.
%   length_limit : controls segment length.
%                 - If 0 < length_limit <= 1: segment length is a random fraction of the boundary length,
%                   uniformly sampled from [0.1, length_limit].
%                 - If length_limit > 1: returns the full boundary as the segment.
%                 Default: 0.25
%
% Outputs
%   selectedSegment : uint8 [L x 2] array of (row, col) boundary points for the selected segment
%   startIndex      : starting index (1-based) along the closed boundary polyline
%
% Notes
% - The boundary coordinates follow bwboundaries convention: (row, col).
% - The returned segment is contiguous along the polyline and wraps around if needed.
%

% ----------------------------
% Default arguments
% ----------------------------
if ~exist('length_limit', 'var') || isempty(length_limit)
    length_limit = 0.25;
end

validateattributes(length_limit, {'numeric'}, {'scalar', '>', 0});

% ----------------------------
% Ensure binary foreground image
% ----------------------------
bwImage = bwImage ~= 0;

% ----------------------------
% Extract boundaries (ignore holes)
% ----------------------------
boundaries = bwboundaries(bwImage, 'noholes');

if isempty(boundaries)
    error('selectBoundary:NoBoundary', 'No boundary found in the input binary image.');
end

% ----------------------------
% Select the longest boundary as the primary geometry
% ----------------------------
[~, idx] = max(cellfun(@length, boundaries));
boundary = boundaries{idx};

if size(boundary, 1) < 3
    error('selectBoundary:BoundaryTooShort', 'Extracted boundary is too short to sample a segment.');
end

% ----------------------------
% Ensure the boundary is closed (first point equals last point)
% ----------------------------
if ~isequal(boundary(1, :), boundary(end, :))
    boundary = [boundary; boundary(1, :)];
end

% Use the closed polyline length for sampling. Note: last point duplicates the first.
totalPoints = size(boundary, 1);

% ----------------------------
% Determine segment length
% ----------------------------
if length_limit > 1
    % Compatibility with the original implementation: return the full boundary.
    segmentLength = totalPoints;
else
    % Random fraction in [0.1, length_limit] (same as original logic)
    frac = rand * (length_limit - 0.1) + 0.1;
    segmentLength = ceil(frac * totalPoints);
end

% Safety clamp: ensure at least 2 points in the segment
segmentLength = max(segmentLength, 2);
segmentLength = min(segmentLength, totalPoints);

% ----------------------------
% Randomly choose the starting index
% ----------------------------
startIndex = randi(totalPoints);

% ----------------------------
% Extract contiguous segment (with wrap-around)
% ----------------------------
endIndex = startIndex + segmentLength - 1;

if endIndex <= totalPoints
    selectedSegment = boundary(startIndex:endIndex, :);
else
    wrapAround = endIndex - totalPoints;
    selectedSegment = [boundary(startIndex:end, :); boundary(1:wrapAround, :)];
end

selectedSegment = uint8(selectedSegment);

end
