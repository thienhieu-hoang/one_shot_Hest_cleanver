To generate DeepMIMOv2/Static_BS16/TransmitedGrid.mat:
    - Run eepMIMOv2/gen_XGrid.m
To generate DeepMIMOv2/Static_BS16/pilot_value.mat:
    - Run DeepMIMOv2/LoopSNR_ .m, set breakpoint after pilot_tx(pilot_Indices) = pilot_Symbols;
        then save 'pillot_tx' 612x14 matrix
                  'pilots_idx' 408x2 (rows,column) idx