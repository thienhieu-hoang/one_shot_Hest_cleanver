% function to generate txWaveform
function [txWaveform, carrier, dmrsSymbols, dmrsIndices] = wave_gen()
    carrier = nrCarrierConfig;
    carrier.NSizeGrid = 51;            % Bandwidth in number of resource blocks (51 RBs at 30 kHz SCS for 20 MHz BW)
    carrier.SubcarrierSpacing = 30;    % 15, 30, 60, 120, 240 (kHz)
    carrier.CyclicPrefix = "Normal";   % "Normal" or "Extended" (Extended CP is relevant for 60 kHz SCS only)
    carrier.NCellID = 2;

    % PDSCH and DM-RS configuration
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
    pdsch.DMRS.DMRSPortSet = 0:pdsch.NumLayers-1; % DM-RS ports to use for the layers
    pdsch.DMRS.DMRSTypeAPosition = 2;      % Mapping type A only. First DM-RS symbol position (2,3)
    pdsch.DMRS.DMRSLength = 1;             % Number of front-loaded DM-RS symbols (1(single symbol),2(double symbol))
    pdsch.DMRS.DMRSAdditionalPosition = 1; % Additional DM-RS symbol positions (max range 0...3)
    pdsch.DMRS.DMRSConfigurationType = 2;  % DM-RS configuration type (1,2)
    pdsch.DMRS.NumCDMGroupsWithoutData = 1;% Number of CDM groups without data
    pdsch.DMRS.NIDNSCID = 1;               % Scrambling identity (0...65535)
    pdsch.DMRS.NSCID = 0;                  % Scrambling initialization (0,1)

    % Generate DM-RS indices and symbols
    dmrsSymbols = nrPDSCHDMRS(carrier,pdsch);
    dmrsIndices = nrPDSCHDMRSIndices(carrier,pdsch);
    
    % Create resource grid
    pdschGrid = nrResourceGrid(carrier);
    
    % Map PDSCH DM-RS symbols to the grid
    pdschGrid(dmrsIndices) = dmrsSymbols;
    
    % OFDM-modulate associated resource elements
    txWaveform = nrOFDMModulate(carrier,pdschGrid);
end