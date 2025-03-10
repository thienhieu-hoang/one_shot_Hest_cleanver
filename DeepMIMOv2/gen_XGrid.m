nSamples = 1e4;
X = zeros(14, 612, 2, nSamples);


    % Parameters (assumed from your dataset_params)
    dataset_params.OFDM.NSizeGrid = 51;      % 51 RBs
    dataset_params.OFDM.delta_f = 30;        % 30 kHz
    dataset_params.OFDM.num_symbol = 14;     % 14 symbols per slot
    
    % Create Resource Grid
    pilot_tx = zeros(dataset_params.OFDM.NSizeGrid*12, dataset_params.OFDM.num_symbol); % 612 x 14
    
    % Carrier configuration
    carrier = nrCarrierConfig;
    carrier.NSizeGrid = dataset_params.OFDM.NSizeGrid;          % 51 RBs
    carrier.SubcarrierSpacing = dataset_params.OFDM.delta_f;    % 30 kHz
    carrier.CyclicPrefix = "Normal";
    carrier.NCellID = 2;
    % carrier.SymbolsPerSlot = dataset_params.OFDM.num_symbol;  % Read-only
    
    % PDSCH configuration
    pdsch = nrPDSCHConfig;
    pdsch.PRBSet = 0:carrier.NSizeGrid-1; % Full band allocation
    pdsch.SymbolAllocation = [0, carrier.SymbolsPerSlot];
    pdsch.MappingType = "A";
    pdsch.NID = carrier.NCellID;
    pdsch.RNTI = 1;
    pdsch.VRBToPRBInterleaving = 0;
    pdsch.NumLayers = 1;
    pdsch.Modulation = "16QAM";  % Modulation is 16-QAM
    
    % DM-RS configuration
    pdsch.DMRS.DMRSPortSet = 0:pdsch.NumLayers-1;
    pdsch.DMRS.DMRSTypeAPosition = 2;
    pdsch.DMRS.DMRSLength = 1;
    pdsch.DMRS.DMRSAdditionalPosition = 1;
    pdsch.DMRS.DMRSConfigurationType = 2;
    pdsch.DMRS.NumCDMGroupsWithoutData = 1;
    pdsch.DMRS.NIDNSCID = 1;
    pdsch.DMRS.NSCID = 0;
    
    % Get DM-RS indices and symbols
    pilot_Indices = nrPDSCHDMRSIndices(carrier, pdsch);
    pilot_Symbols = nrPDSCHDMRS(carrier, pdsch);
    
    % Map DM-RS to grid
    pilot_tx(pilot_Indices) = pilot_Symbols;  % Pilot grid with DM-RS
    
    % Generate PDSCH data
    [pdsch_indices, pdsch_info] = nrPDSCHIndices(carrier, pdsch);  % PDSCH indices
    num_bits = pdsch_info.G;  % Bit capacity
    
    
    % Create full transmit grid
    tx_grid = zeros(dataset_params.OFDM.NSizeGrid*12, dataset_params.OFDM.num_symbol); % 612 x 14
    tx_grid(pilot_Indices) = pilot_Symbols;  % Map DM-RS # equals to pilot_tx above
    bit_stream = zeros(num_bits, nSamples);

for n = 1:nSamples
    bit_stream(:, n) = randi([0 1], num_bits, 1);  % Random bit stream
    pdsch_symbols = nrPDSCH(carrier, pdsch, bit_stream(:, n));  % 16-QAM symbols
    tx_grid(pdsch_indices) = pdsch_symbols;  % Map PDSCH data
    
    x_real_imag = cat(3, real(tx_grid), imag(tx_grid)); % Concatenate real and imaginary parts along the 3rd dimension
    X(:,:,:,n) = reshape(permute(x_real_imag, [2, 1, 3]), [14, 612, 2, 1]);

    if ~mod(n,50)
        if n>50
            fprintf(repmat('\b',1,lineLength))
        end
        lineLength = fprintf(['Finished ', num2str(n), '/', num2str(nSamples)]);
    end       
end

save("DeepMIMO_Data\TransmitedGrid.mat", "X", "bit_stream", "dataset_params", "carrier", "pdsch", '-v7.3')