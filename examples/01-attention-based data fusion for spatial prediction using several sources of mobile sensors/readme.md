Attention-based data fusion for spatial prediction using several sources of mobile sensors
============================================================================================

Based on a synthetic spatial phenomena, and several synthetic sensors of different quality, we will test four types of attention-based data fusion.

All of them are Multi-dimension Attention with Distance as Score (MADS):
- kriging w/ weight described as a constant over space
- Nadaraya-Watson with distance as score w/ weight described as a constant over space
- kriging w/ weight described as NN, so non-constant over space
- Nadaraya-Watson with distance as score w/ weight described as NN, so non-constant over space

This example gather four parts (data construction and three cases studies):

0) Construction of the synthetic sensors (noted as 0a, 0b, 0c)

1) Both the mobiles sensors for the training and the target are of high quality (noted as 1a,1b,1c,1d)

2) The mobiles sensors for the training are of low quality and the target of high quality (noted as 2a,2b,2c,2d)

3)  The mobiles sensors for the training are of six different qualities and the target of high quality (noted as 3a,3b,3c,3d)
