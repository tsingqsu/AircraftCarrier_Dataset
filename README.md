# AircraftCarrier-Dataset

This repository is to download the AircraftCarrier Dataset

Author: KAI KANG, GANGMING PANG, XUN ZHAO, JIABAO WANG, YAN LI.

Last Update: 20/01/2020

CITATION:

If you use AircraftCarrier Dataset in your research, please cite:

	@PROCEEDINGS {AircraftCarrier Dataset,
	author       = "KAI KANG, GANGMING PANG, XUN ZHAO, JIABAO WANG, YAN LI",
	title        = "AircraftCarrier Dataset: A New Benchmark for Instance-level Image Classification",
	year         = "2020",
	organization = "***"
	}
	
This document is to explain the use of AircraftCarrier Dataset and its metadata.

## 1.Copyright Statement

  The AircraftCarrier Dataset are publicly available, you are always free to obtain it. Nonetheless, the AircraftCarrier Dataset in their original resolutions may be subject to copyright, so we do not make them publicly available on the website.
Consequently, if you are a researcher/educator who wish to have a copy of the original images for non-commercial research and/or educational use, we may provide them through E-mail, under certain conditions and at our discretion. The details are as follows:

#### (1) You are asked to agree to and sign the following terms of access.

#### (2) Send an E-mail with your full name, organization and your signature to us,***panggangming16@gmail.com***.

#### (3) We review your request. If we approve it, we will send you an E-mail with an attachment，which is the dataset.

#### (4) You receive the E-mail and download the AircraftCarrier Dataset.

### the terms of access:

  [RESEARCHER_FULLNAME] (the "Researcher") has requested permission to use the AircraftCarrier Dataset. In exchange for such permission, Researcher hereby agrees to the following terms and conditions:
  
  #### Researcher shall use the AircraftCarrier Dataset only for non-commercial research and educational purposes.
  
  #### Researcher accepts full responsibility for his or her use of the AircraftCarrier and shall defend and indemnify us, including their employees, Trustees, officers and agents, against any and all claims arising from Researcher's use of the AircraftCarrier Dataset, including but not limited to Researcher's use of any copies of copyrighted images that he or she may create from the AircraftCarrier Dataset.
  
  #### Researcher may provide research associates and colleagues with access to the Database provided that they first agree to be bound by these terms and conditions.
  
  #### If Researcher is employed by a for-profit, commercial entity, Researcher's employer shall also be bound by these terms and conditions, and Researcher hereby represents that he or she is fully authorized to enter into this agreement on behalf of such employer.
  
  #### We make no representations or warranties regarding the AircraftCarrier Dataset, including but not limited to warranties of non-infringement or fitness for a particular purpose.
  
  #### We reserve the right to terminate Researcher's access to the AircraftCarrier Dataset at any time.
  
  If you agree to the terms above, we will send you an E-mail with an attachment.
  
## 2. Dataset Introduction
  
  ***Notes: All of the codes we support work well dependent on Python3.6 & Pytorch 1.1.***
  
  Our experiments were performed on a deep learning workstation with 3.5GHz Intel Core E5-2637v4 CPU, Nvidia GTX 1080Ti GPU with 11GB RAM and Ubuntu 16.04 operating system. The programming environment is based on Python language, PyCharm integrated development environment and Pytorch deep learning toolkit are used by GPU accelerated training.The details are showed in the paper.
  
### This project includes series of files as follows:

  ### ‘InstanceCLS_train.py’
  
    Run it to train the net.
    
  ### ‘InstanceCLS_test.py’
  
    Run it to get the classification accuracy results on AircraftCarrier dataset with different methods. 
    
  ### Models
  
    We provide six widely used models as follows:
    
    --DenseNet121
    
    --MobileNet_V2
    
    --ResNet50
    
    --ShuffleNet_V2
    
    --SqueezeNet_V2
    
    --VGGNet16
    
    See `models/__init__.py` for details regarding what keys to use to call these models
    
  ### results
  
    It shows the classification accuracy results on AircraftCarrier dataset with different methods. It also consists of the labels of the test dataset.
    
    Eg.1 The ‘squeezenetv2_confusion_matirx.png’ shows the confusion matrix of the classification of the model SqueezeNet_V2. 
    
    Eg.2 The ‘densenet121_pred_label.txt’ shows the labels what the model DenseNet121 predicted.
    
  ### untils
  
    It shows the process to divide the images into two datasets, train dataset and test dataset, load the images and caculate the losses. 
    
    It also consists of the losses, optimzer and predictor etc.
    
    

