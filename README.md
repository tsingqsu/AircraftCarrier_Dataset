# AircraftCarrier_Dataset

This repository is to download the AircraftCarrier dataset

Author: Kai Kang, Gangming Pang, Xun Zhao, Jiabao Wang, Yang Li.

Last Update: 26/01/2020

CITATION:

If you use AircraftCarrier dataset in your research, please cite:

	@ARTICLE {AircraftCarrier_Dataset,
	author       = "Kai Kang, Gangming Pang, Xun Zhao, Jiabao Wang, Yang Li",
	title        = "A New Benchmark for Instance-level Image Classification",
	year         = "2020",
	}

## 1.Dataset Download

The AircraftCarrier Dataset are publicly available, if you agree to the terms of its usage, you can obtain it by sending us a scanned [Applicant Form](https://github.com/tsingqsu/AircraftCarrier_Dataset/tree/master/info/Application_Form.docx) to ***panggangming16@gmail.com***. We will send you an e-mail with a download link.

Note that we make no representations or warranties regarding the AircraftCarrier dataset, including but not limited to warranties of non-infringement or fitness for a particular purpose.

## 2. Dataset Introduction

The dataset contains the following 20 categories: ``Liaoning (CV16)'', ``Cavour (CVH550)'', ``Giuseppe Garibaldi (CVH551)'', ``USS Nimitz (CVN68)'', ``USS Dwight David Eisenhower (CVN69)'', ``USS Carl Vinson (CVN70)'', ``USS Theodore Roosevelt (CVN71)'', ``USS Abraham Lincoln (CVN72)'', ``USS George Washington (CVN73)'', ``USS John C. Stennis (CVN74)'', ``USS Harry S. Truman (CVN75)'', ``USS Ronald Reagan (CVN76)'', ``USS George H.W. Bush (CVN77)'', ``USS Gerald R. Ford (CVN78)'', ``Juan Carlos I (L61)'', ``HMS Queen Elizabeth (R08)'', ``INS Viraat (R22)'', ``Charles de Gaulle (R91)'', ``HTMS Chakri Naruebet (R911)'' and ``Admiral Flota Sovetskogo Soyuza Kuznetsov (RN063)''. The following Figure presents examples of the 20 aircraft carriers.
<img align="right" src="https://github.com/tsingqsu/AircraftCarrier_Dataset/info/fig_2.png">
&nbsp;
&nbsp;

## 3. Code Usage

***Notes: All of the codes we support work well dependent on Python3.6 & Pytorch 1.1.***

Before running the code, you should download the dataset and modify the dataset path.
  ### Training
   ```Shell
   ./InstanceCLS_train.sh
   ```
  ### Testing
   ```python
   python InstanceCLS_test.py
   ```
