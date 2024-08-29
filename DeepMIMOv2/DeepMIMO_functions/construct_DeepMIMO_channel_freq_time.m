% --------- DeepMIMO: A Generic Dataset for mmWave and massive MIMO ------%
% Authors: Ahmed Alkhateeb, Umut Demirhan, Abdelrahman Taha 
% Date: March 17, 2022
% Goal: Encouraging research on ML/DL for MIMO applications and
% providing a benchmarking tool for the developed algorithms
% ---------------------------------------------------------------------- %

% Output:
%  ORIGINAL CODE:  channel == M_Rx x M_Tx x subcs  complex
%                             M_UE x M_BS x subcs
%  channel == 1 x M_BS cell (M_BS is the antenna number of BS t)
%           channel{m_bs} == subcs x 14 complex
%                         ==   612 x 14
function [channel] = construct_DeepMIMO_channel_freq_time(tx_ant_size, tx_rotation, tx_ant_spacing, rx_ant_size, rx_rotation, rx_ant_spacing, params_user, params)
    % m_bs == the index of considering antenna in tx_ant_size of BS
BW = params.bandwidth; % Hz
ang_conv = pi/180;
Ts = 1/BW;
% Ts = 1/30720000;
k=(0:params.OFDM_sampling_factor:params.OFDM_limit-1).';
num_sampled_subcarriers=length(k);

% TX antenna parameters for a UPA structure
M_TX_ind = antenna_channel_map(tx_ant_size(1), tx_ant_size(2), tx_ant_size(3), 0); % M_TX x 3
M_TX =prod(tx_ant_size); % == M_BS
kd_TX=2*pi*tx_ant_spacing;

% RX antenna parameters for a UPA structure
M_RX_ind = antenna_channel_map(rx_ant_size(1), rx_ant_size(2), rx_ant_size(3), 0); % M_RX x 3 
M_RX=prod(rx_ant_size);
kd_RX=2*pi*rx_ant_spacing;

if params_user.num_paths == 0
    channel = complex(zeros(M_RX, M_TX, num_sampled_subcarriers));
    return
end


% Change the DoD and DoA angles based on the panel orientations
if params.activate_array_rotation
    [DoD_theta, DoD_phi, DoA_theta, DoA_phi] = axes_rotation(tx_rotation, params_user.DoD_theta, params_user.DoD_phi, rx_rotation, params_user.DoA_theta, params_user.DoA_phi);
else
    DoD_theta = params_user.DoD_theta;
    DoD_phi = params_user.DoD_phi;
    DoA_theta = params_user.DoA_theta;
    DoA_phi = params_user.DoA_phi;
end

% Apply the radiation pattern of choice
if params.radiation_pattern % Half-wave dipole radiation pattern
    power = params_user.power.* antenna_pattern_halfwavedipole(DoD_theta, DoD_phi) .* antenna_pattern_halfwavedipole(DoA_theta, DoA_phi);
else % Isotropic radiation pattern
    power = params_user.power;
end

% TX Array Response - BS
gamma_TX=+1j*kd_TX*[sind(DoD_theta).*cosd(DoD_phi);
              sind(DoD_theta).*sind(DoD_phi);
              cosd(DoD_theta)]; % == 3 x L
array_response_TX = exp(M_TX_ind*gamma_TX); % == M_TX x L 

% RX Array Response - UE or BS
gamma_RX=+1j*kd_RX*[sind(DoA_theta).*cosd(DoA_phi);
                    sind(DoA_theta).*sind(DoA_phi);
                    cosd(DoA_theta)];       % 3 x L
array_response_RX = exp(M_RX_ind*gamma_RX); % == M_RX x L (L=no_paths)


channel = cell(1,M_TX);

%Assuming the pulse shaping as a dirac delta function and no receive LPF
if ~params.activate_RX_filter
    % % Account only for the channel within the useful OFDM symbol duration
    % delay_normalized = (params_user.ToA + (ofdm_sym_idx-1)* params.OFDM.T_symbol)/Ts;
    %                         % [second] 
    
    delay_normalized = (params_user.ToA)/Ts;
    if params.staticChan == 1
        % path_const=sqrt(power/params.num_OFDM).*exp(1j*params_user.phase*ang_conv).*exp(-1j*2*pi*(k/params.num_OFDM)*delay_normalized);
        path_const=sqrt(power/params.num_OFDM).*exp(1j*params_user.phase*ang_conv).*exp(-1j*2*pi*(k/params.num_OFDM)*delay_normalized);
                 %      1 x no_paths          .*            1 x no_paths         .* exp(         subcs x 1          * 1 x no_paths) 
                 %      subc x no_paths (no_paths = L)
        channel_temp = sum(reshape(array_response_RX, M_RX, 1, 1, []) .* reshape(array_response_TX, 1, M_TX, 1, []) .* reshape(path_const, 1, 1, num_sampled_subcarriers, []), 4);
                 %     sum( M_UE x 1 x 1 x L                          .*  1 x M_BS x 1 x L                          .* 1 x 1 x subcs x L                                     , 4) 
                 %     sum( M_UE x M_BS x subcs x L, 4)
                 %     M_UE x M_BS x subcs
        % channel_temp == M_Rx x M_Tx x subcs  complex
                           % M_UE x M_BS x subcs
                           %    1 x M_BS x subcs
        channel_temp_1 = permute(channel_temp, [3,2,1]); 
                           % subc x M_BS x 1

        for m_bs = 1:M_TX 
            channel{m_bs} = repmat(channel_temp_1(:, m_bs), 1, params.OFDM.num_symbol); % 612 x 14
        end
    else
        for ofdm_sym_idx = 1:(params.OFDM.num_symbol) % 1:14

            % Doppler Effect
            f_D = params.maximumDopplerShift * rand(1, params_user.num_paths);
                % 1 x L
            Doppler_effect = exp( 1j * pi* f_D * params.OFDM.T_symbol) .* sin(pi * f_D * params.OFDM.T_symbol) ./ (pi * f_D.* params.OFDM.T_symbol) ;
                % 1 x L 

            % power(delay_normalized >= params.num_OFDM) = 0;
            % delay_normalized(delay_normalized>=params.num_OFDM) = params.num_OFDM;
            
            % path_const=sqrt(power/params.num_OFDM).*exp(1j*params_user.phase*ang_conv).*exp(-1j*2*pi*(k/params.num_OFDM)*delay_normalized);
            path_const=sqrt(power/params.num_OFDM).*exp(1j*params_user.phase*ang_conv).*exp(-1j*2*pi*(k/params.num_OFDM)*delay_normalized) .* Doppler_effect;
                     %      1 x no_paths          .*            1 x no_paths         .* exp(         subcs x 1          * 1 x no_paths) 
                     %      subc x no_paths (no_paths = L)
            channel_temp = sum(reshape(array_response_RX, M_RX, 1, 1, []) .* reshape(array_response_TX, 1, M_TX, 1, []) .* reshape(path_const, 1, 1, num_sampled_subcarriers, []), 4);
                     %     sum( M_UE x 1 x 1 x L                          .*  1 x M_BS x 1 x L                          .* 1 x 1 x subcs x L                                     , 4) 
                     %     sum( M_UE x M_BS x subcs x L, 4)
                     %     M_UE x M_BS x subcs
            % channel_temp == M_Rx x M_Tx x subcs  complex
                               % M_UE x M_BS x subcs
                               %    1 x M_BS x subcs
            channel_temp_1 = permute(channel_temp, [3,2,1]); 
                               % subc x M_BS x 1
    
            for m_bs = 1:M_TX 
                channel{m_bs}(:, ofdm_sym_idx) = channel_temp_1(:, m_bs);
            end
        end
    end
end

end