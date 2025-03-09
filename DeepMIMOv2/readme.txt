Files to generate resource grid 64x14 with 14 varied channels for 14 OFDM symbols:
    - GenerateData_Main_freq_time.m
    - parameters_freq_time.m
    - DeepMIMO_function/construct_DeepMIMO_channel_freq_time.m
    - DeepMIMO_function/DeepMIMO_generator_freq_time.m

change the file index '_ver1', '_ver2',... before generating data

To generate resource grid 64x14 with static channel (14 same channels for 14 OFDM symbols):
    - set params.staticChan == 1 in parameters_freq_time.m

To generate DeepMIMOv2/Static_BS16/TransmitedGrid.mat:
    - Run eepMIMOv2/gen_XGrid.m
To generate DeepMIMOv2/Static_BS16/pilot_value.mat:
    - Run DeepMIMOv2/LoopSNR_ .m, set breakpoint after pilot_tx(pilot_Indices) = pilot_Symbols;
        then save 'pillot_tx' 612x14 matrix
                  'pilots_idx' 408x2 (rows,column) idx