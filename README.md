# HighFedelity2D

This code is for high fedelity electronic structure prediction (including bandgap, valence band maximun, and conduction band minimun) on HSE06 level for 2D material in order to enable accurate band alignmemt of 2D herostructure, especially for solar cell application. This model alleviates the data scarcity problem by effectively incorporating 2D dataset of lower fedelity on GGA level and the larger 3D dataset. The data used for the model training is avaliable at [dropbox](https://www.dropbox.com/scl/fi/h0n2uh655x5ga0s7utkle/bandstructure_data.tar.gz?rlkey=1orxb2p313feqzftay5ip8q88&st=yjcbjb5w&dl=0), please place the extracted subdirectories under .../data.



Here is the model performance for HSE level electronic structure prediction:

|      Method    | Bandgap (eV, HSE06) | VBM (eV, HSE06)| CBM (eV, HSE06) | Stability (cross entropy) |
|----------------|---------------------|----------------|-----------------|---------------------------|
| from scratch   |       0.384         |     0.344      |      0.328      |         0.52              |
| multi-fidelity |       0.168         |     0.143      |      0.119      |         0.32              |


The model structure and the comparison between predicted HSE result and ground truth result are as follows:

<!-- ![Model Structure](figures/Figure%206.png) -->

<img src="figures/Figure%206.png" width="80%">