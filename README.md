# AttSwinUNet For Echocardiogram Segmentation
Implementation code for [Attention Swin U-Net: Cross-Contextual Attention Mechanism for Skin Lesion Segmentation](https://arxiv.org/abs/2210.16898) paper for segmentation of LV in Echocardiograms. The dataset used is [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/)

## Environment and Installation

- Please prepare an environment with python=3.7, and then use the command below to install all the dependencies
```pip install -r requirements.txt```

---
## Running the code
### Training

```bash
 python train.py
```

### Validation
```bash
python eval.py
```

- Modify hyperparameters in the code itself

---
## References
- [AttSwinUNet](https://github.com/NITR098/AttSwinUNet)
- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
- [CrossViT](https://github.com/IBM/CrossViT)
---
## Citation
```
@article{aghdam2022attention,
  title={Attention Swin U-Net: Cross-Contextual Attention Mechanism for Skin Lesion Segmentation},
  author={Aghdam, Ehsan Khodapanah and Azad, Reza and Zarvani, Maral and Merhof, Dorit},
  journal={arXiv preprint arXiv:2210.16898},
  year={2022}
}
```
