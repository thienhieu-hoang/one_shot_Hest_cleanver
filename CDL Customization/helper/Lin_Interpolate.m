function [H_equalized, H_linear] = Lin_Interpolate(Y_noise, pilot_Indices, pilot_Symbols)
% Perform linear interpolation of the grid and input the result to the
% neural network This helper function extracts the DM-RS symbols from
% dmrsIndices locations in the received grid rxGrid and performs linear
% interpolation on the extracted pilots.

    % Obtain pilot symbol estimates
    dmrsRx = Y_noise(pilot_Indices);
    dmrsEsts = dmrsRx .* conj(pilot_Symbols);

    % Create empty grids to fill after linear interpolation
    [H_equalized, rxDMRSGrid, H_linear] = deal(zeros(size(Y_noise)));
    rxDMRSGrid(pilot_Indices)  = pilot_Symbols;
    H_equalized(pilot_Indices) = dmrsEsts;
    
    % Find the row and column coordinates for a given DMRS configuration
    [rows,cols] = find(rxDMRSGrid ~= 0);
    dmrsSubs = [rows,cols,ones(size(cols))];
    [l_hest,k_hest] = meshgrid(1:size(H_linear,2),1:size(H_linear,1));

    % Perform linear interpolation
    f = scatteredInterpolant(dmrsSubs(:,2),dmrsSubs(:,1),dmrsEsts);
    H_linear = f(l_hest,k_hest);

end