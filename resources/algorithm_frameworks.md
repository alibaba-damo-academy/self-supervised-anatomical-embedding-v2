# Frameworks

### SAM

See [arXiv](https://arxiv.org/abs/2012.02383). Flowchart:

![](./SAM_framework1.png)

Network and loss:

![](./SAM_framework2.png)

### SAM++

See [arXiv](https://arxiv.org/abs/2306.13988). Semantic head:

![](./framework-a-SAMpp.png)

Structural inference (fixed-point iteration):

![](./framework-b-SAMpp.png)

### Cross-SAM

Cross-SAM is a generalized SAM framework for cross-modality embedding learning, see [arXiv](https://arxiv.org/abs/2307.03535).

Flowchart:

![](./framework-CSAM.png)

MRI to CT point matching results:

![](./examples.png)

Cross-SAM can be used for cross-modality registration, even if the two modalities have large FOV difference, see the 
following examples with large CT images and small MR ones. 

We first use Aggressive-SAM/Cross-SAM to find grid matching points
on the CT-MRI data, and then do rigid registration, followed by DEEDS deformable fine-tuning.

The Aggressive-SAM is trained using only intra-modality data and would result some false alignment but the overall body part is correct.
Cross-SAM is trained on top of the registration results of the Aggressive-SAM. 

![](./regis_result.png)
