# Additive-Manufacturing-Sensor-Selection
Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in LPBF Process


# Journal link
https://doi.org/10.1016/j.jmatprotec.2023.118144

# Overview

Additive Manufacturing (AM) has revolutionized manufacturing processes, offering unprecedented flexibility in creating intricate geometries while minimizing material waste through near-net shape manufacturing. Laser Powder Bed Fusion (LPBF) within AM has gained prominence for its ability to produce complex components. However, the demand for extending LPBF's capabilities to multi-material printing, expanding design possibilities and eliminating additional assembly steps is growing. This expansion presents process control and quality monitoring challenges, especially for ensuring precise material composition. This study investigates Acoustic Emission (AE) signals and optical emissions in the context of laser wavelength during LPBF processes involving multiple materials. Experimental data were collected from five distinct powder compositions using a custom monitoring system integrated into a commercial LPBF machine. By employing contrastive deep learning techniques, this research effectively categorized signals arising when processing different powder compositions. Notably, prediction accuracy significantly improved when combining AE and optical data and training Convolutional Neural Networks (CNNs) using contrastive learning. Integrating contrastive learning with a sensor fusion strategy played a crucial role in monitoring LPBF processes involving multiple materials. Visualizations of the CNN models' lower-level embedded space, trained with two contrastive loss functions, showcased the capacity to cluster acoustic and optical emissions according to their similarities, aligning with the five different compositions. This research advances our understanding of Multi-material LPBF processes and underscores the potential of AE and optical sensors in enhancing quality control and material composition monitoring in AM. 

![Picture1](https://github.com/vigneashpandiyan/Additive-Manufacturing-Multi-Material-Composition-Monitoring-Using-Sensor-Fusion/assets/39007209/6993fa32-9934-4fb3-8dfb-e172b02c52b1)


# Methodology
In summary, this research has centered on the analysis of AE signals and the intensity of optical emissions within the context of laser wavelength for monitoring LPBF processes involving multiple materials. The investigation involved the examination of five distinct powder compositions, with experimental data collected using a custom monitoring system integrated into a commercial LPBF machine. The culmination of our experiments and analyses has provided valuable insights into the LPBF process, illuminating the relationships among powder composition, AE signals, and optical emissions. Throughout this study, we applied methodologies grounded in contrastive deep learning to oversee and categorize signals associated with different powder compositions effectively. As we reflect on the outcomes, several overarching conclusions become apparent:

•	The AE waveform signal was divided into fifteen evenly spaced frequency bands from 0Hz to 150 kHz, revealing significant differences in energy levels among the five powder compositions, signifying varying melt pool dynamics. Moreover, a noticeable correlation was observed between the intensity of optical signals at the laser wavelength and the powder composition, even when using identical process parameters.
•	When performing supervised classification with a CNN on both the photodiode signals and acoustic data, the prediction accuracy was found to be lower than when employing a strategy that fused both types of information and trained a CNN with a similar architecture.
•	Combining a contrastive learner with a sensor fusion strategy has proven to be valuable for monitoring the LPBF process involving multiple materials. When visualizing the lower-level embedded space of the CNN models trained using the two contrastive loss functions, it became evident that the multivariate data streams corresponding to the five compositions could be grouped together based on their similarities. 
•	Using learned representations from two CNN models as input for XGBoost-based classification yielded higher accuracy than conventional CNN training. This highlights the potential of feature representations from contrastive learning methods for monitoring tasks. 

![image](https://github.com/vigneashpandiyan/Additive-Manufacturing-Transfer-Learning/assets/39007209/de11305c-119f-4269-b271-8a4847f59e1c)


# Code
```bash
git clone https://github.com/vigneashpandiyan/Additive-Manufacturing-Multi-Material-Composition-Monitoring-Using-Sensor-Fusion
cd Additive-Manufacturing-Multi-Material-Composition-Monitoring-Using-Sensor-Fusion
python Main.py
```

# Citation
```
@article{PANDIYAN2023118144,
title = {Optimizing In-situ Monitoring for Laser Powder Bed Fusion Process: Deciphering Acoustic Emission and Sensor Sensitivity with Explainable Machine Learning},
journal = {Journal of Materials Processing Technology},
pages = {118144},
year = {2023},
issn = {0924-0136},
doi = {https://doi.org/10.1016/j.jmatprotec.2023.118144},
url = {https://www.sciencedirect.com/science/article/pii/S0924013623002893},
author = {Vigneashwara Pandiyan and Rafał Wróbel and Christian Leinenbach and Sergey Shevchik},
keywords = {Laser Powder Bed Fusion, Process Monitoring, Empirical Mode Decomposition, Acoustic Emission, Explainable AI (XAI)},
}
