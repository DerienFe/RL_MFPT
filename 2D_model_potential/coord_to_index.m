%index converting function from x-y cartesian to matlab 1-indexed matrix
%for x,y in (-3, 3) N = 13.
%TW 1st June 2023

function [idx_i, idx_j] = coord_to_index(coord_i, coord_j)
    % Convert 2D grid coordinates to indices
    % Input: 
    %   coord_i, coord_j: coordinates on the grid
    %   ngrid: number of grid points in each dimension
    % Output:
    %   idx_i, idx_j: corresponding indices in the array

    % Calculate the spacing between states
    spacing = 0.5;

    % Convert coordinates to indices
    idx_i = round(1 + (coord_i + 3) / spacing);
    idx_j = round(1 + (coord_j + 3) / spacing);
end