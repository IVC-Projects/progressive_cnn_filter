# An RDO-Free CNN In-Loop Filtering Approach For Inter Frame Coding
Dandan Ding, Lingyi Kong, Fengqing Zhu<br>

---

## Abstract
Convolutional Neural Network (CNN) has been introduced
to in-loop filtering in video coding for further performance
improvement. Various CNN structures have been designed. These
CNN models are usually trained through learning the correlations
between the reconstructed and the original frames and then applied
to every single reconstructed frame to improve the video quality.
Such a direct model training and deployment strategy is effective
for intra coding but will obtain only a locally optimal model and
hence trigger an over-filtering problem in inter coding because
the intertwined reference dependencies across inter frames are not
taken into account. To address this issue, state-of-the-art technologies
usually resort to the Rate-Distortion Optimization (RDO) or the
skipping method to apply the CNN model only to selective coding
blocks or frames. However, such schemes cannot fundamentally solve
the problem because the local CNN model is inaccurate. In this
paper, we present an RDO-free approach to coordinate the CNNbased
in-loop filters to work seamlessly with video encoders. 

---

## Requirements
+ python 3.7
+ tensorflow >=1.6.0 && <=2.0.0
+ Visual studio >=2013
+ HM 16.9
+ AOM 1.0.0

---

## Usage
### Trainning 
+ Trainning database : [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
+ Trainning method
  +  Each frame of DIV2K is encoded using HM16.9 and libaom v1.0.0 with the setting of all loop filter off to obtain the raw reconstructed frames.
  +  Loss function: $f\left ( \Theta  \right )= \frac{1}{K}\sum_{k=1}^{K}\left \| f_{cnn}\left ( X_k;\Theta - Y_k \right ) \right \|_{2}^{2}$
  +  When a model is obtained, the image enhanced with this model is added to the training set to continue training, and Repeat again and again ...
### Testing 
+ HEVC
  + TEncGop.cpp
+ AV1
  + encode.c

---


## Model
                             _____     _____           _____
               raw          | 3x3 |   | 3x3 |         | 3x3 |    ___
           reconstructed--->| 64  |-->| 64  |-->...-->| 64  |-->|add|-->filtered
               frame     |  |_____|   |_____|         |_____|    —^—
                         |______________shotcut___________________|

---

## Experiments and Results

### 1、Average bitrate (kbps) and psnr (db) of using different cnnn in inter in-loop filtering

|Model|LDP||||||||RA||||||||
|:---:|:---------:|:----:|:-----------------------:|:---------------------:|:--------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
||QP=37||QP=32||QP=27||QP=22||QP=37||QP=32||QP=27||QP=22||
||bitrate|PSNR|bitrate|PSNR|bitrate|PSNR|bitrate|PSNR|bitrate|PSNR|bitrate|PSNR|bitrate|PSNR|bitrate|PSNR|
|anchor|923.794|31.806|1860.973|34.495|4246.250|37.278|12540.883|40.295|951.230|32.317|1823.547|34.900|3816.174|37.522|9841.134|40.190|
|CNN1|927.707|32.098|1840.493|34.765|4162.843|37.511|12442.042|40.496|945.974|32.731|1792.072|35.249|3743.491|37.794|9706.375|40.385|
|CNN2|921.478|32.148|1843.724|34.763|4162.424|37.519|**12459.105**|**40.494**|942.131|32.763|1792.041|35.256|**3744.338**|**37.799**|**9701.845**|**40.385**|
|CNN5|921.106|32.162|**1833.963**|**34.798**|**4160.810**|**37.535**|12480.928|40.490|**941.990**|**32.773**|**1791.046**|**35.262**|3745.463|37.802|9711.090|40.379|
|CNN8|**920.610**|**32.171**|–|–|4161.774|37.518|–|–|942.081|32.774|–|–|–|–|–|–|

<br>

### 2、Cnn models used in h.265/hevc experiments. here cnn0 represents the direct model

|QP values|LDP|RA|
|:---:|:---------:|:----:|
|37|CNN8|CNN5|
|32|CNN5|CNN5|
|27|CNN5|CNN2|
|22|CNN0/CNN2|CNN0/CNN2|

<br>

### 3、BD-RATE of our proposed approach compared with previous solutions

|Class|Sequence|AI||LDP||||RA|||
|:---:|:---------:|:----:|:-----------------------:|:---------------------:|:--------:|:-------:|:--------:|:--------:|:---------:|:--------:|
|||AI|frame skipping|CU skipping|CTU_RDO|propsed|fram skipping|CU skipping|CTU_RDO|propsed|
|A|PeopleOnStreet|-9:55%|-3:88%|-2:74%|-5:24%|-6:50%|-3:24%|-6:53%|-7:21%|-8:14%|
||Traffic|-10:80%|-7:52%|+5:99%|-7:47%|-8:90%|-7:38%|-6:37%|-9:54%|-10:99%|
|B|BasketballDrive|-8:58%|-5:83%|-5:80%|-9:78%|-9:82%|-3:36%|-6:35%|-8:13%|-8:70%|
||BQTerrace|-5:72%|-9:86%|-5:79%|-12:27%|-10:77%|-8:65%|-9:30%|-12:27%|-11:67%|
||Cactus|-7:76%|-7:17%|+2:49%|-8:72%|-8:46%|-6:07%|-6:45%|-10:14%|-9:80%|
||Kimono|-8:40%|-4:23%|-3:84%|-5:82%|-6:43%|-2:16%|-5:41%|-5:74%|-6:08%|
||ParkScene|-8:32%|-3:33%|+2:77%|-3:62%|-4:67%|-3:55%|-4:46%|-6:28%|-7:28%|
|C|BQMall|-10:41%|-7:63%|-4:62%|-9:37%|-10:48%|-6:27%|-7:51%|-9:65%|-10:69%|
||PartyScene|-6:44%|-3:82%|-1:12%|-5:88%|-5:96%|-3:05%|-3:74%|-5:98%|-6:40%|
||BasketballDrill|-15:78%|-9:02%|+2:22%|-11:45%|-11:75%|-7:62%|-4:75%|-11:17%|-11:96%|
||RaceHorsesC|-6:14%|-2:94%|-2:71%|-4:98%|-4:90%|-2:18%|-4:86%|-5:80%|-5:84%|
|D|BasketballPass|-11:54%|-7:41%|-7:37%|-9:46%|-10:69%|-5:76%|-8:00%|-8:94%|-10:24%|
||BlowingBubbles|-8:54%|-4:49%|-0:56%|-5:67%|-6:07%|-3:53%|-4:14%|-6:36%|-7:39%|
||BQSquare|-8:43%|-8:38%|-5:85%|-9:66%|-11:09%|-6:65%|-6:57%|-8:43%|-9:59%|
||RaceHorses|-10:70%|-4:46%|-5:18%|-7:02%|-7:98%|-3:13%|-7:21%|-7:87%|-8:45%|
|E|Johnny|-13:61%|-16:13%|+2:16%|-16:68%|-17:95%|-12:59%|-10:85%|-14:64%|-16:88%|
||FourPeople|-13:95%|-14:02%|-2:56%|-14:79%|-15:47%|-12:85%|-12:23%|-14:96%|-16:38%|
||KristenAndSara|-13:07%|-13:47%|-2:61%|-14:14%|-15:20%|-11:51%|-11:70%|-13:76%|-15:64%|
||Class A|-10:18%|-5:70%|+1:63%|-6:36%|-7:70%|-5:31%|-6:45%|-8:37%|-9:56%|
||Class B|-7:75%|-6:09%|-2:03%|-8:04%|-8:03%|-4:76%|-6:39%|-8:51%|-8:71%|
||Class C|-9:69%|-5:85%|-1:56%|-7:92%|-8:27%|-4:78%|-5:21%|-8:15%|-8:72%|
||Class D|-9:80%|-6:19%|-4:74%|-7:95%|-8:96%|-4:77%|-6:48%|-7:90%|-8:92%|
||Class E|-13:54%|-14:54%|-1:00%|-15:21%|-16:21%|-12:32%|-11:59%|-14:45%|-16:30%|
||Average|**-9:87%**|-7:42%|-1:95%|-9:00%|**-9:62%**|-6:09%|-7:02%|-9:27%|**-10:12%**|




  
