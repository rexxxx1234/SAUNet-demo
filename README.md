# SAUNet: Shape Attentive U-Net for Interpretable Medical Image Segmentation

**Abstract**: Medical image segmentation is a difficult but important task for many clinical operations such as cardiac bi-ventricular volume estimation. More recently, there has been a shift to utilizing deep learning and fully convolutional neural networks (CNNs) to perform image segmentation that has yielded state-of-the-art results in many public benchmark datasets. Despite the progress of deep learning in medical image segmentation, standard CNNs are still not fully adopted in clinical settings as they lack robustness and interpretability. Shapes are generally more meaningful features than solely textures of images, which are features regular CNNs learn, causing a lack of robustness. Likewise, previous works surrounding model interpretability have been focused on post hoc gradient-based saliency methods. However, gradient-based saliency methods typically require additional computations post hoc and have been shown to be unreliable for interpretability. Thus, we present a new architecture called Shape Attentive U-Net (SAUNet) which focuses on model interpretability and robustness. The proposed architecture attempts to address these limitations by the use of a secondary shape stream that captures rich shape-dependent information in parallel with the regular texture stream. Furthermore, we suggest multi-resolution saliency maps can be learned using our dual-attention decoder module which allows for multi-level interpretability and mitigates the need for additional computations post hoc. Our method also achieves state-of-the-art results on the two large public cardiac MRI image segmentation datasets of SUN09 and AC17.

**Architecture**: 
![Architecture](https://github.com/rexxxx1234/SAUNet-demo/blob/master/Screen%20Shot%202020-03-26%20at%201.42.46%20PM.png)
![Attention Block](https://github.com/rexxxx1234/SAUNet-demo/blob/master/block.png)

**Link**: https://arxiv.org/pdf/2001.07645.pdf

If you find our work helpful, please consider citing our work: 

```
@misc{sun2020saunet,
    title={SAUNet: Shape Attentive U-Net for Interpretable Medical Image Segmentation},
    author={Jesse Sun and Fatemeh Darbeha and Mark Zaidi and Bo Wang},
    year={2020},
    eprint={2001.07645},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```
