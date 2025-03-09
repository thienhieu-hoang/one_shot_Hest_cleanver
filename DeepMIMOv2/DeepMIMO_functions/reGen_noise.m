% this file is to regenerate the noise that the 'data-generating Loop file'
% ("LoopSNR_v...") forgot to save the Noise

cd(fileparts(mfilename('fullpath'))); % switches to the directory of this script being executed.

addpath('..')
domain = 'mapBased';
if domain == 'DeepMIMO'
    data_folder = 'DeepMIMO_Data\Static_BS16\freq_symb_1ant_612sub_ver4';
elseif domain == 'mapBased'
    data_folder = '..\..\CDL Customization\Data\ver39_';
end

row = '3500_3516';
for snr = -25:5:30
    if domain == 'DeepMIMO'
        filename = [data_folder, '\Gan_', num2str(snr), '_dBOutdoor1_3p4_1ant_612subcs_Row_', row, '.mat'];
    elseif domain == 'mapBased'
        filename = [data_folder, '\', num2str(snr), 'dB\1_mapBaseData.mat'];
    end
    load(filename);


    % -------------- Regenerate DM-RS --------------------- %
    % Load Dataset Parameters
    dataset_params = read_params('parameters_freq_time.m');
    dataset_params.staticChan = 1; % set to 1 to generate 612x14 channel grid, all 14 channels of 14 OFDM symbols are the same
    % run parameters.m
    
    % Settings 
    bs_ant = prod(dataset_params.num_ant_BS(1,:)); % M = 1x4x1 BS Antennas
    subs = dataset_params.OFDM_limit; % subcarriers
    % pilot_l = 16; % 8; % Pilots length is 8
    
    % Generate Pilots 
    % create Resource Grid
    pilot_tx = zeros(dataset_params.OFDM.NSizeGrid*12, dataset_params.OFDM.num_symbol); % 612 x 14
    
    carrier = nrCarrierConfig;
        carrier.NSizeGrid = dataset_params.OFDM.NSizeGrid;          % 51;            % Bandwidth in number of resource blocks (51 RBs at 30 kHz SCS for 20 MHz BW)
        carrier.SubcarrierSpacing = dataset_params.OFDM.delta_f;    % 30;    % 15, 30, 60, 120, 240 (kHz)
        carrier.CyclicPrefix = "Normal";                            % "Normal" or "Extended" (Extended CP is relevant for 60 kHz SCS only)
        carrier.NCellID = 2;                                        % Cell identity
        % carrier.SymbolsPerSlot = dataset_params.OFDM.num_symbol;  % (read-only) 14
    
    pdsch = nrPDSCHConfig;
        pdsch.PRBSet = 0:carrier.NSizeGrid-1; % PDSCH PRB allocation
        pdsch.SymbolAllocation = [0, carrier.SymbolsPerSlot];           % PDSCH symbol allocation in each slot
        pdsch.MappingType = "A";     % PDSCH mapping type ("A"(slot-wise),"B"(non slot-wise))
        pdsch.NID = carrier.NCellID;
        pdsch.RNTI = 1;
        pdsch.VRBToPRBInterleaving = 0; % Disable interleaved resource mapping
        pdsch.NumLayers = 1;            % Number of PDSCH transmission layers
        pdsch.Modulation = "16QAM";                       % "QPSK", "16QAM", "64QAM", "256QAM"
    
        % DM-RS configuration
        pdsch.DMRS.DMRSPortSet = 0: pdsch.NumLayers-1; % DM-RS ports to use for the layers
        pdsch.DMRS.DMRSTypeAPosition = 2;      % Mapping type A only. First DM-RS symbol position (2,3)
        pdsch.DMRS.DMRSLength = 1;             % Number of front-loaded DM-RS symbols (1(single symbol),2(double symbol))
        pdsch.DMRS.DMRSAdditionalPosition = 1; % Additional DM-RS symbol positions (max range 0...3)
        pdsch.DMRS.DMRSConfigurationType = 2;  % DM-RS configuration type (1,2)
        pdsch.DMRS.NumCDMGroupsWithoutData = 1;% Number of CDM groups without data
        pdsch.DMRS.NIDNSCID = 1;               % Scrambling identity (0...65535)
        pdsch.DMRS.NSCID = 0;                  % Scrambling initialization (0,1)
    
    % get indices of pilots in Resource Grid
    pilot_Indices = nrPDSCHDMRSIndices(carrier,pdsch);
    
    % generate pilot symbols
    pilot_Symbols = nrPDSCHDMRS(carrier,pdsch);
    
    % map into Resource Grid to generate pilot_tx
    pilot_tx(pilot_Indices) = pilot_Symbols; % == subc x sym == 612 x 14


    x_real_imag = cat(3, real(pilot_tx), imag(pilot_tx)); % Concatenate real and imaginary parts along the 3rd dimension
    x_reshaped = reshape(permute(x_real_imag, [2, 1, 3]), [14, 612, 2, 1]);
    
    Y = H_data.*x_reshaped;
    
    Noise = Y_data -Y; % 14 x 612 x 2 x Nsamples

    save(filename, 'Noise', '-append');
end
