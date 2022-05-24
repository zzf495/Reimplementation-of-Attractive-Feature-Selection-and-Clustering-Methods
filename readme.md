

# The Re-implementation of Feature Selection & Clustering Methods

This repository integrates the codes for some **feature selection & clustering** methods. I know how hard it is to reproduce the codes, especially for beginners, and I hope it can help.

- If you find any errors or need any help in reproducing the code, please feel free to contact me. 

- If any author or publisher has questions, please contact me to remove or replace them.

- My email is coding495@163.com

---

**To run supervised method**:

> 'Demo_Supervised.m' gives a simple example for supervised methods. 
>
> To run the codes, the size of the inputs are: <img src="https://latex.codecogs.com/svg.image?X_1\in&space;\mathbb{R}^{m\times&space;n_1},&space;X_2\in&space;\mathbb{R}^{m\times&space;n_2},&space;Y_1\in&space;\mathbb{R}^{n_1\times&space;1},&space;Y_2\in&space;\mathbb{R}^{n_2\times&space;1}" />, where *m* is the dimension, and <img src="http://latex.codecogs.com/svg.latex?n_1" title="http://latex.codecogs.com/svg.latex?n_1" /> and <img src="http://latex.codecogs.com/svg.latex?n_2" title="http://latex.codecogs.com/svg.latex?n_2" /> present the number of the training and test samples, respectively (<img src="http://latex.codecogs.com/svg.latex?Y_2" title="http://latex.codecogs.com/svg.latex?Y_2" /> is used to calculate the clustering results, and is not involved in training).

**To run unsupervised method**:

> 'Demo_Unsupervised.m' gives a simple example for unsupervised methods. 
>
> To run the codes, the size of the inputs are: <img src="https://latex.codecogs.com/svg.image?X\in&space;\mathbb{R}^{m\times&space;n},&space;Y\in&space;\mathbb{R}^{n\times&space;1}" />, where *m* is the dimension, and <img src="http://latex.codecogs.com/svg.latex?n" title="http://latex.codecogs.com/svg.latex?n" /> presents the number of the training samples (<img src="http://latex.codecogs.com/svg.latex?Y" title="http://latex.codecogs.com/svg.latex?Y" /> is used to calculate the clustering results, and is not involved in training).

## The codes exist in the repository

### Feature Selection

#### - Supervised Methods

- 2019-RSLDA [[7]](https://ieeexplore.ieee.org/abstract/document/8272002): Robust Sparse Linear Discriminant Analysis.

- 2021-DSDPL [[8]](https://www.sciencedirect.com/science/article/pii/S0031320320303848): Dual subspace discriminative projection learning

  > I achieves 90% accuracy on `AR_Face` with 560-dimensions, while the authors adopt 540-dimensions for experiments. The results of the codes may not be consistent with the paper.

---

#### - Unsupervised Methods

- 2019-LRLMR [[1]](https://www.sciencedirect.com/science/article/pii/S0893608019301212): Unsupervised feature selection via latent representation learning and manifold regularization.

  > We reproduce the codes as same as the descriptions of the paper. The official codes are available at http://tangchang.net/

- 2019-URAFS [[2]](https://ieeexplore.ieee.org/abstract/document/8474999): Generalized Uncorrelated Regression with Adaptive Graph for Unsupervised Feature Selection.

- 2021-AGUFS [[3]](https://www.sciencedirect.com/science/article/pii/S0950705121004196): Adaptive graph-based generalized regression model for unsupervised feature selection.

- 2021-DSLRL [[4]](https://www.sciencedirect.com/science/article/pii/S0031320321000601): Dual space latent representation learning for unsupervised feature selection.

- 2022-DLUFS [[5]](https://www.sciencedirect.com/science/article/pii/S0957417422005437): Low-rank dictionary learning for unsupervised feature selection.

  > The official codes (python implementation) are available at https://github.com/mohsengh/DLUFS/

- 2022-SLMEA [[6]](https://www.sciencedirect.com/science/article/pii/S0925231222001916): Sparse and low-dimensional representation with maximum entropy adaptive graph for feature selection.

---

## Clustering & Other

>  The codes will be available soon.

# Reference

[1] Tang, Chang, et al. "Unsupervised feature selection via latent representation learning and manifold regularization." *Neural Networks* 117 (2019): 163-178.

[2] X. Li, H. Zhang, R. Zhang, Y. Liu and F. Nie, "Generalized Uncorrelated Regression with Adaptive Graph for Unsupervised Feature Selection," in IEEE Transactions on Neural Networks and Learning Systems, vol. 30, no. 5, pp. 1587-1595, May 2019, doi: 10.1109/TNNLS.2018.2868847.

[3] Huang, Yanyong, et al. "Adaptive graph-based generalized regression model for unsupervised feature selection." *Knowledge-Based Systems* 227 (2021), doi: 10.1016/j.knosys.2021.107156.

[4] Shang, Ronghua, et al. "Dual space latent representation learning for unsupervised feature selection." *Pattern Recognition* 114 (2021), doi: 10.1016/j.patcog.2021.107873.

[5] Parsa, Mohsen Ghassemi, Hadi Zare, and Mehdi Ghatee. "Low-rank dictionary learning for unsupervised feature selection." *Expert Systems with Applications* 202 (2022), doi: 10.1016/j.eswa.2022.117149.

[6] Shang, Ronghua, et al. "Sparse and low-dimensional representation with maximum entropy adaptive graph for feature selection." *Neurocomputing* 485 (2022): 57-73.

[7] J. Wen et al., "Robust Sparse Linear Discriminant Analysis," in IEEE Transactions on Circuits and Systems for Video Technology, vol. 29, no. 2, pp. 390-403, Feb. 2019, doi: 10.1109/TCSVT.2018.2799214.

[8] Belous, Gregg, Andrew Busch, and Yongsheng Gao. "Dual subspace discriminative projection learning." *Pattern Recognition* 111 (2021), doi: 10.1016/j.patcog.2020.107581.