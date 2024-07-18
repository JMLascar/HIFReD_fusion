**Hyperspectral Image Fusion via Regularized Deconvolution — the HIFReD fusion algorithm.**

With the recent launch of the X-Ray Imaging and Spectroscopy Mission (XRISM) and the advent of microcalorimeter detectors, X-ray astrophysics is entering in a new era of spatially resolved high resolution spectroscopy.
But while this new generation of X-ray telescopes have much finer spectral resolutions than their predecessors (e.g. XMM-Newton, Chandra), they also have coarser spatial resolutions, leading to problematic cross-pixel contamination. This issue is currently a critical limitation for the study of extended sources such as galaxy clusters of supernova remnants.
To increase the scientific output of XRISM's hyperspectral data, we propose to fuse it with XMM-Newton data, and seek to obtain a cube with the best spatial and spectral resolution of both generations. 
This is the aim of hyperspectral fusion. 

With HIFReD, we have implemented an algorithm that jointly deconvolves the spatial response of XRISM and the spectral response of XMM-Newton. To do so, we have constructed a forward model adapted for instrumental systematic degradations and Poisson noise, which is shown in this diagram: 

![image](https://github.com/user-attachments/assets/80bcbfcb-28e1-4399-a1d2-a35ba28361da)

We tackle hyperspectral fusion as a regularized inverse problem. Let $X$ be the data set with better spatial resolution (e.g. XMM-Newton) and $Y$ the data set with better spectral resolution (e.g. XRISM). Let $\hat{Z}$ be the superresolute cube we are looking for. The equation we seek to minimize is then: 
<p>
$$
\hat{Z}= \text{argmin}_{Z\geq 0} \mathcal{L}_P\Big(X \;| \;Z_X\Big)+\mathcal{L}_P\Big(Y \;| \;Z_Y\Big) + \varphi(Z),
$$
</p>
where $\mathcal{L}_P$ is the negative Poisson log-likelihood, $\varphi$ is the regularisation term, and:
<p>
$Z_X=(t_{X} Z\odot A_{X})\mathcal{S}_{X} \mathcal{R}_L $
</p><p>
$Z_Y=\mathcal{R}_{MN}\mathcal{B}_Y(t_YZ\odot A_Y)$
</p>

$t$ are the exposure times, $A$ are the effective areas, $\mathcal{B}$ $_Y$ is the operator that convolves each slice of Z with the PSF of $Y$, $\mathcal{S}$ $_X$ is the operator that convolves each spectra of Z with the spectral response of $X$, and $\mathcal{R}$ are the rebinning operators. 


In this code, we provide the three methods of regularization tested in LASCAR J., BOBIN J., ACERO F. (2024) : spectral low rank approximation with a spatial Sobolev regularization; spectral low rank approximation with a 2D wavelt sparsity constraint ; and a 2D-1D wavelet sparsity constraint. We also provide an option for low rank with no spatial regularization, and no regularization at all. 

The jupyter notebook HIFReD_Example.ipynb provides a list of the necessary packages, and an example of how to use HIFReD on the provided toy models. These models are in the directory "toy_models", and are the same as those being presented in the article. The directory "code" has the HIFReD.py file, along with a file for the starlet transform (pyStarlet_master_2D1D.py) needed for the wavelet regularization. 
