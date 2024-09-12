% Generate data, X, Y, H in the same size  % subc x symbol x noUE x M_BS
% ----------------- Add the path of DeepMIMO function --------------------%
addpath('DeepMIMO_functions')

% -------------------- DeepMIMO Dataset Generation -----------------------%
% Load Dataset Parameters
dataset_params = read_params('parameters_freq_time.m');
dataset_params.staticChan = 1; % set to 1 to generate 612x14 channel grid, all 14 channels of 14 OFDM symbols are the same
% run parameters.m

%% Settings 
bs_ant = prod(dataset_params.num_ant_BS(1,:)); % M = 1x4x1 BS Antennas
subs = dataset_params.OFDM_limit; % subcarriers
% pilot_l = 16; % 8; % Pilots length is 8

%% Generate Pilots 
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

SNR = 5:5:30; %10:5:35;
Rows = [3500 3516]; %[550 568; 3540 3595; 5150 5160];
for snr = SNR
    for r = 1:size(Rows, 1)
        dataset_params.active_user_first = Rows(r,1);
        dataset_params.active_user_last  = Rows(r,2);
        filename = ['Outdoor1_3p4_',num2str(bs_ant),'ant_',num2str(subs),'subcs_Row_', num2str(dataset_params.active_user_first),'_', num2str(dataset_params.active_user_last)];
    
        %% Generate channel dataset H                            
        % dataset_params.saveDataset = 1;
         
        % -------------------------- Dataset Generation -----------------%
        [DeepMIMO_dataset, dataset_params] = DeepMIMO_generator_freq_time(dataset_params); % Get H (i.e.,DeepMIMO_dataset )

        % Output: 
            %   DeepMIMO_dataset == 1x1 cell == 1 x (number of active BSs) cell
            %       DeepMIMO_dataset{1} == 1x1 struct 
            %           DeepMIMO_dataset{1}.user == 1x noUE cell (total number of UEs = 2211)
            %               each cell: DeepMIMO_dataset{1}.user{1} == 1x1 struct, with attributes: 
            %                   DeepMIMO_dataset{1}.user{1}.channel == 1xM_BS cell
            %                       DeepMIMO_dataset{1}.user{1}.channel{m_bs} == subc x num_OFDMsymbol (612 x 14)
            %                                                           m_bs is the antenna index of BS 1
            %                                                           
            %                                                           
            %                       DeepMIMO_dataset{1}.user{1}.loc == 1x3 double
            %   dataset_params : add 2 attributes: 
            %       dataset_params.num_BS   ()
            %       dataset_params.num_user ()
        
        
        %% Generate Signal Y with Noise
        % channels = zeros(subs, dataset_params.OFDM.num_symbol, length(DeepMIMO_dataset{1}.user), bs_ant); % subc x symbol x noUE x M_BS 
        % Y        = zeros(subs, dataset_params.OFDM.num_symbol, length(DeepMIMO_dataset{1}.user), bs_ant); % subc x symbol x noUE x M_BS 
        % Y_noise  = zeros(subs, dataset_params.OFDM.num_symbol, length(DeepMIMO_dataset{1}.user), bs_ant); % subc x symbol x noUE x M_BS 
        % H_equalized = zeros(subs, dataset_params.OFDM.num_symbol, length(DeepMIMO_dataset{1}.user), bs_ant); % subc x symbol x noUE x M_BS 
        % H_linear = zeros(subs, dataset_params.OFDM.num_symbol, length(DeepMIMO_dataset{1}.user), bs_ant); % subc x symbol x noUE x M_BS 
        % H_practical = zeros(subs, dataset_params.OFDM.num_symbol, length(DeepMIMO_dataset{1}.user), bs_ant); % subc x symbol x noUE x M_BS 
        channels    = [];
        Y           = [];
        Y_noise     = [];
        H_equalized = [];
        H_linear    = [];
        H_practical = [];
        
        X_data = [];
        H_equalized_data = [];
        H_linear_data = [];
        H_practical_data = [];
        Y_data = [];
        % true channel data
        H_data = [];
        
        
        % 1-bit ADC
        % Y_sign   = zeros(subs, dataset_params.OFDM.num_symbol, length(DeepMIMO_dataset{1}.user), bs_ant);  % noUE x M_BS x pilot x 2
        
        idx = 0; 
        for i = 1:length(DeepMIMO_dataset{1}.user)
            if iscell(DeepMIMO_dataset{1}.user{i}.channel) % not cell: that user doesnt have channel
                idx = idx +1; 
                for anten_bs = 1: bs_ant
                    temp_H_equalized = zeros(subs, dataset_params.OFDM.num_symbol);
            
                    %-- channels(:,:,i,anten_bs) = normalize(DeepMIMO_dataset{1}.user{i}.channel{anten_bs},'scale');    % subc x symbol           
                    channels(:,:,idx,anten_bs) = DeepMIMO_dataset{1}.user{i}.channel{anten_bs};    % subc x symbol
        
                    Y(:,:,idx,anten_bs)        = DeepMIMO_dataset{1}.user{i}.channel{anten_bs} .* pilot_tx;           % subc x symbol
                    Y_noise(:,:,idx,anten_bs)  = awgn(Y(:,:,idx,anten_bs),snr,'measured');                                       % subc x symbol
                                                % is it correct to add noise in freq domain?
        
                    [H_equalized(:,:,idx,anten_bs), H_linear(:,:,idx,anten_bs)] = Lin_Interpolate(Y_noise(:,:,idx,anten_bs), pilot_Indices, pilot_Symbols); 
        
                    H_practical(:,:,idx,anten_bs) = nrChannelEstimate(Y_noise(:,:,idx,anten_bs), pilot_Indices, pilot_Symbols);
                    
        
                    %append just non zeros H
                end
            end
        end
        
        X_data = pilot_tx;
        
        H_equalized_data(:,:,:,:,1) = real(H_equalized);
        H_equalized_data(:,:,:,:,2) = imag(H_equalized);
        
        H_linear_data(:,:,:,:,1) = real(H_linear);
        H_linear_data(:,:,:,:,2) = imag(H_linear);
        
        H_practical_data(:,:,:,:,1) = real(H_practical);
        H_practical_data(:,:,:,:,2) = imag(H_practical);
        
        Y_data(:,:,:,:,1) = real(Y_noise); % subc x symbol x noUE x M_BS
        Y_data(:,:,:,:,2) = imag(Y_noise); % subc x symbol x noUE x M_BS
        % use complex(A,B) to return
        %  x M_BS
        
        % true channel data
        H_data(:,:,:,:,1) = real(channels);
        H_data(:,:,:,:,2) = imag(channels);
        % M_BS x subcs
        
        %% Convert complex data to two-channel data
        % 1-bit ADC?
        % Y_sign(:,:,:,:,1) = sign(real(Y_noise)); % real part of Y
        % Y_sign(:,:,:,:,2) = sign(imag(Y_noise)); % imag part of Y
        % 
        % 
        % channels_r(:,:,:,1) = real(channels); % real part of H
        % channels_r(:,:,:,2) = imag(channels); % imag part of H
        
        % Shuffle data 
        % shuff            = randi([1,size(channels,3)], size(channels,3), 1);  % random pick, not permute
        shuff            = randperm(size(channels,3))';  % permute shuffle

        H_equalized_data = H_equalized_data(:,:,shuff,:,:);
        H_linear_data    = H_linear_data(:,:,shuff,:,:);
        H_practical_data = H_practical_data(:,:,shuff,:,:);
        Y_data           = Y_data(:,:,shuff,:,:);
        H_data           = H_data(:,:,shuff,:,:); % subc x symbol x noUE x M_BS x 2
        
        H_equalized_data = permute(H_equalized_data, [2,1,5,3,4]);
        H_linear_data    = permute(H_linear_data, [2,1,5,3,4]);
        H_practical_data = permute(H_practical_data, [2,1,5,3,4]);
        Y_data           = permute(Y_data, [2,1,5,3,4]);
        H_data           = permute(H_data, [2,1,5,3,4]);
            % symb x subc x 2 x samples x 1 (BS_ant) 
            % to get size in python
            % samples (noUE) x 2 x subc x symb (size in Pytorch)
        
        
        % %% Save data
        if dataset_params.staticChan == 1
            % Check if the save folder exists
            % save_file = ['DeepMIMO_Data/Static_BS16/freq_symb_',num2str(bs_ant),'ant_',num2str(subs),'sub_ver4/Gan_',num2str(snr),'_dB',filename];
            save_folder = ['DeepMIMO_Data/Static_BS16/freq_symb_',num2str(bs_ant),'ant_',num2str(subs),'sub_ver4/Gan_',num2str(snr),'_dB',filename];
            if ~exist(save_folder, 'dir')
                mkdir(save_folder);
            end
            save(save_folder,'H_data', 'H_linear_data', 'H_equalized_data', 'Y_data', 'H_practical_data',"dataset_params",'-v7.3');
        end
    end
end

% helper function
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
