Files to generate resource grid 64x14 with 14 varied channels for 14 OFDM symbols:
    - GenerateData_Main_freq_time.m
    - parameters_freq_time.m
    - DeepMIMO_function/construct_DeepMIMO_channel_freq_time.m
    - DeepMIMO_function/DeepMIMO_generator_freq_time.m

To generate resource grid 64x14 with static channel (14 same channels for 14 OFDM symbols):
    - set params.staticChan == 1 in parameters_freq_time.m
