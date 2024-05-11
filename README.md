# Additive-Manufacturing-Multi-Material-Composition-Monitoring-Using-Sensor-Fusion
Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in LPBF Process


# Journal link
https://doi.org/10.1016/j.jmatprotec.2023.118144

# Overview

Additive Manufacturing (AM) has revolutionized manufacturing processes, offering unprecedented flexibility in creating intricate geometries while minimizing material waste through near-net shape manufacturing. Laser Powder Bed Fusion (LPBF) within AM has gained prominence for its ability to produce complex components. However, the demand for extending LPBF's capabilities to multi-material printing, expanding design possibilities and eliminating additional assembly steps is growing. This expansion presents process control and quality monitoring challenges, especially for ensuring precise material composition. This study investigates Acoustic Emission (AE) signals and optical emissions in the context of laser wavelength during LPBF processes involving multiple materials. Experimental data were collected from five distinct powder compositions using a custom monitoring system integrated into a commercial LPBF machine. By employing contrastive deep learning techniques, this research effectively categorized signals arising when processing different powder compositions. Notably, prediction accuracy significantly improved when combining AE and optical data and training Convolutional Neural Networks (CNNs) using contrastive learning. Integrating contrastive learning with a sensor fusion strategy played a crucial role in monitoring LPBF processes involving multiple materials. Visualizations of the CNN models' lower-level embedded space, trained with two contrastive loss functions, showcased the capacity to cluster acoustic and optical emissions according to their similarities, aligning with the five different compositions. This research advances our understanding of Multi-material LPBF processes and underscores the potential of AE and optical sensors in enhancing quality control and material composition monitoring in AM. 

![Picture1](https://github.com/vigneashpandiyan/Additive-Manufacturing-Multi-Material-Composition-Monitoring-Using-Sensor-Fusion/assets/39007209/6993fa32-9934-4fb3-8dfb-e172b02c52b1)


# Methodology
In summary, this research has centered on the analysis of AE signals and the intensity of optical emissions within the context of laser wavelength for monitoring LPBF processes involving multiple materials. The investigation involved the examination of five distinct powder compositions, with experimental data collected using a custom monitoring system integrated into a commercial LPBF machine. The culmination of our experiments and analyses has provided valuable insights into the LPBF process, illuminating the relationships among powder composition, AE signals, and optical emissions. Throughout this study, we applied methodologies grounded in contrastive deep learning to oversee and categorize signals associated with different powder compositions effectively. As we reflect on the outcomes, several overarching conclusions become apparent. Our methodology incorporates multiple sensor modalities, including acoustic and optical intensities in the laser wavelength spectrum. Specifically, our work explores the challenges associated with monitoring multi-material deposition, particularly in materials such as copper that exhibit high reflectivity to infrared wavelengths when combined with stainless steel. This aspect presents a novel and significant challenge, which we aim to address through data-driven approaches and optimization of sensor fusion techniques. This paper builds upon existing research by integrating data fusion techniques, facilitating the exchange of complementary information among these sensors and thereby enhancing their correlation with actual flaws in compositional monitoring. This advancement lays the groundwork for achieving high confidence in quality control in future LPBF applications involving multi-materials.

The AE waveform signal was divided into fifteen evenly spaced frequency bands from 0Hz to 150 kHz, revealing significant differences in energy levels among the five powder compositions, signifying varying melt pool dynamics. Moreover, a noticeable correlation was observed between the intensity of optical signals at the laser wavelength and the powder composition, even when using identical process parameters. When performing supervised classification with a CNN on both the photodiode signals and acoustic data, the prediction accuracy was found to be lower than when employing a strategy that fused both types of information and trained a CNN with a similar architecture.	Combining a contrastive learner with a sensor fusion strategy has proven to be valuable for monitoring the LPBF process involving multiple materials. When visualizing the lower-level embedded space of the CNN models trained using the two contrastive loss functions, it became evident that the multivariate data streams corresponding to the five compositions could be grouped together based on their similarities. Using learned representations from two CNN models as input for XGBoost-based classification yielded higher accuracy than conventional CNN training. This highlights the potential of feature representations from contrastive learning methods for monitoring tasks. 

![image](https://github.com/vigneashpandiyan/Additive-Manufacturing-Multi-Material-Composition-Monitoring-Using-Sensor-Fusion/assets/39007209/e4834045-02e1-491e-81e3-2f4d62a7c3c3)


# Code
```bash
git clone https://github.com/vigneashpandiyan/Additive-Manufacturing-Multi-Material-Composition-Monitoring-Using-Sensor-Fusion
cd Additive-Manufacturing-Multi-Material-Composition-Monitoring-Using-Sensor-Fusion
python Main.py
```

# Citation
```
@article{
title = {Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in LPBF Process},
journal = {Virtual and Physical Prototyping},
year = {2024},
author = {Vigneashwara Pandiyan and Antonios Baganis  and Roland Axel Richter and Rafał Wróbel and Christian Leinenbach},
keywords = {Laser Powder Bed Fusion, Multi-material Process Monitoring, Acoustic Emission, Optical Emission, Contrastive Learning)},
}
