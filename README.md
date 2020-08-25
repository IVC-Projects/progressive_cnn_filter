# A Frame-level CNN-based In-Loop Filter For Inter Frame Coding
Dandan Ding, Lingyi Kong, Wenyu Wang, Fengqing Zhu<br>

---

## Abstract
Various Convolutional Neural Network (CNN) structures have been designed for in-loop filtering in video coding which showed performance improvement. These CNN models are usually trained through learning the correlations between the reconstructed and the original frames, which are then applied to every single reconstructed frame to improve the overall video quality. Such a direct model training and deployment strategy is effective for intra coding but will obtain only a locally optimal model. This triggers an over-filtering problem in inter coding because the intertwined reference dependencies across inter frames are not taken into account. To address this issue, state-of-the-art methods usually resort to the Rate-Distortion Optimization (RDO) so that the CNN model only applies to selective coding blocks or frames. However, such schemes cannot fundamentally solve the problem because the direct CNN model is inaccurate.In this paper, we propose a new approach to train and coordinate CNN-based in-loop filters to work seamlessly with video encoders.
### Examples showing the subjective quality of frames suffering from the over-fifiltering problem.The fifirst line shows the uncompressed frames and the second illustrates the over-fifiltered frames.

![](https://github.com/IVC-Projects/progressive_cnn_filter/blob/master/pictures/picture.png)

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
+ Trainning dataset
  +  We use [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset for CNN training. Each frame of DIV2K is encoded using H.265/HEVC reference software HM16.9 to obtain the raw reconstructed frames. At the encoder, we use the default AI configuration “encoder intra main.cfg” of HM16.9, except that the traditional in-loop filters including Deblocking and SAO are turned off. 
+ Training settings. 
  +  Frames are segmented into 64×64 patches as samples and the batch size is set to 64. We adopt the Adaptive moment estimation (Adam) algorithm for stochastic optimization. To train the direct model CNN0, the initial learning rate is set as 10^-4^. For the transfer learning phase of models CNNi (0 < i ≤ N), the initiallearning rate is set as 10^-5^.During training, the learning rate is adjusted using the step strategy with γ = 0.5. 
  +  Loss function: $f\left ( \Theta  \right )= \frac{1}{K}\sum_{k=1}^{K}\left \| f_{cnn}\left ( X_k;\Theta - Y_k \right ) \right \|_{2}^{2}$
  +  When a model is obtained, the image enhanced with this model is added to the training set to continue training, and Repeat again and again ...
### Testing 
+ Testing settings
  + We use 18 test sequences which are mostly selected by the Joint Collaborative Team on Video Coding (JCT-VC) to evaluate the video coding efficiency.The first 50 frames of each sequence are used for evaluation. In H.265/HEVC, we follow the default LDP configuration “encoder lowdelay main.cfg” and RA configuration “encoder randomaccess main.cfg” with Deblocking and SAO off.
  
---
  
## Model
                             _____     _____           _____
               raw          | 3x3 |   | 3x3 |         | 3x3 |    ___
           reconstructed--->| 64  |-->| 64  |-->...-->| 64  |-->|add|-->filtered
               frame     |  |_____|   |_____|         |_____|    —^—
                         |______________shotcut___________________|

---

## Experiments and Results

### 1、Overall performance

Convergence of CNN~N~. First, we need to find out when the progressive CNN model converges. Experiments are conducted to determine the value of N, the number of times to tune the direct CNN model, so as to terminate the progressive training. Here the direct model is termed as CNN~0~ for clarification. 

We have the following observations.

+ LDP requires a higher N than RA.
+ N increases as QP value increases
+ Certain frames are insensitive to the over-filtering effect.

#### <p align="center">Table 1: Average bitrate (kbps) and psnr (db) of using different cnnn in inter in-loop filtering</p>
|Model|LDP||||||||RA||||||||
|:---:|:---------:|:----:|:-----------------------:|:---------------------:|:--------:|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
||QP=37||QP=32||QP=27||QP=22||QP=37||QP=32||QP=27||QP=22||
||bitrate|PSNR|bitrate|PSNR|bitrate|PSNR|bitrate|PSNR|bitrate|PSNR|bitrate|PSNR|bitrate|PSNR|bitrate|PSNR|
|anchor|923.794|31.806|1860.973|34.495|4246.250|37.278|12540.883|40.295|951.230|32.317|1823.547|34.900|3816.174|37.522|9841.134|40.190|
|CNN0|991.619|31.462|1933.679|34.264|4268.964|37.197|12388.495|40.346|901.349|32.152|1708.895|34.784|3554.656|37.495|9176.313|40.246|
|CNN1|927.707|32.098|1840.493|34.765|4162.843|37.511|12442.042|40.496|945.974|32.731|1792.072|35.249|3743.491|37.794|9706.375|40.385|
|CNN2|921.478|32.148|1843.724|34.763|4162.424|37.519|**12459.105**|**40.494**|942.131|32.763|1792.041|35.256|**3744.338**|**37.799**|**9701.845**|**40.385**|
|CNN5|921.106|32.162|**1833.963**|**34.798**|**4160.810**|**37.535**|12480.928|40.490|**941.990**|**32.773**|**1791.046**|**35.262**|3745.463|37.802|9711.090|40.379|
|CNN8|**920.610**|**32.171**|–|–|4161.774|37.518|–|–|942.081|32.774|–|–|–|–|–|–|

<br>

#### Table 2: CNN models used in H.265/HEVC experiments. Here CNN0 represents the direct model
|QP values|LDP|RA|
|:---:|:---------:|:----:|
|37|CNN8|CNN5|
|32|CNN5|CNN5|
|27|CNN5|CNN2|
|22|CNN0/CNN2|CNN0/CNN2|

<br>

### 2、Comparison with existing methods

We compare our approach to existing solutions including the RDO-based method and the skipping method.

>**Note:** the two methods are applied only to inter frames. The intra frames are all filtered by the direct CNN model

Our proposed approach achieves the best performance for all configurations as shown in Table 3.
>In LDP configuration, the frame skipping, CU skipping, and CTU-RDO method achieve 7.42%, 1.95%, and 9.00% BD-rate saving,respectively, whereas our approach gains as much as 9.62%. In RA, the above three methods obtain 6.09%, 7.02%, and 9.27% BD-rate reduction, which is lower than ours at 10.12%.

#### <p align="center">Table 3: BD-rate of our proposed approach compared with previous solutions</p>

|Class|Sequence|AI|LDP|||||RA|||||
|:---:|:---------:|:----:|:-----------------------:|:---------------------:|:--------:|:-------:|:--------:|:--------:|:---------:|:--------:|:--------:|:--------:|
|||AI|Direct use|frame skipping|CU skipping|CTU_RDO|propsed|Direct use|fram skipping|CU skipping|CTU_RDO|propsed|
|A|PeopleOnStreet|-9.55%|+0.98%|-3.88%|-2.74%|-5.24%|-6.50%|-4.51%|-3.24%|-6.53%|-7.21%|-8.14%|
||Traffic|-10.80%|+21.65%|-7.52%|+5.99%|-7.47%|-8.90%|+6.34%|-7.38%|-6.37%|-9.54%|-10.99%|
|B|BasketballDrive|-8.58%|-4.04%|-5.83%|-5.80%|-9.78%|-9.82%|-3.83%|-3.36%|-6.35%|-8.13%|-8.70%|
||BQTerrace|-5.72%|+7.44%|-9.86%|-5.79%|-12.27%|-10.77%|0.43%|-8.65%|-9.30%|-12.27%|-11.67%|
||Cactus|-7.76%|+13.51%|-7.17%|+2.49%|-8.72%|-8.46%|1.22%|-6.07%|-6.45%|-10.14%|-9.80%|
||Kimono|-8.40%|-2.19%|-4.23%|-3.84%|-5.82%|-6.43%|-4.64%|-2.16%|-5.41%|-5.74%|-6.08%|
||ParkScene|-8.32%|+11.08%|-3.33%|+2.77%|-3.62%|-4.67%|3.49%|-3.55%|-4.46%|-6.28%|-7.28%|
|C|BQMall|-10.41%|+2.27%|-7.63%|-4.62%|-9.37%|-10.48%|-1.09%|-6.27%|-7.51%|-9.65%|-10.69%|
||PartyScene|-6.44%|+2.49%|-3.82%|-1.12%|-5.88%|-5.96%|+1.11%|-3.05%|-3.74%|-5.98%|-6.40%|
||BasketballDrill|-15.78%|+10.47%|-9.02%|+2.22%|-11.45%|-11.75%|+2.44%|-7.62%|-4.75%|-11.17%|-11.96%|
||RaceHorsesC|-6.14%|-1.61%|-2.94%|-2.71%|-4.98%|-4.90%|-4.28%|-2.18%|-4.86%|-5.80%|-5.84%|
|D|BasketballPass|-11.54%|+2.92%|-7.41%|-7.37%|-9.46%|-10.69%|-1.05%|-5.76%|-8.00%|-8.94%|-10.24%|
||BlowingBubbles|-8.54%|+3.13%|-4.49%|-0.56%|-5.67%|-6.07%|+0.24%|-3.53%|-4.14%|-6.36%|-7.39%|
||BQSquare|-8.43%|+0.39%|-8.38%|-5.85%|-9.66%|-11.09%|+1.02%|-6.65%|-6.57%|-8.43%|-9.59%|
||RaceHorses|-10.70%|-3.56%|-4.46%|-5.18%|-7.02%|-7.98%|-6.43%|-3.13%|-7.21%|-7.87%|-8.45%|
|E|Johnny|-13.61%|+28.42%|-16.13%|+2.16%|-16.68%|-17.95%|+8.45%|-12.59%|-10.85%|-14.64%|-16.88%|
||FourPeople|-13.95%|+28.52%|-14.02%|-2.56%|-14.79%|-15.47%|+6.38%|-12.85%|-12.23%|-14.96%|-16.38%|
||KristenAndSara|-13.07%|+28.10%|-13.47%|-2.61%|-14.14%|-15.20%|+7.49%|-11.51%|-11.70%|-13.76%|-15.64%|
||Class A|-10.18%|+11.32%|-5.70%|+1.63%|-6.36%|-7.70%|+0.92%|-5.31%|-6.45%|-8.37%|-9.56%|
||Class B|-7.75%|+5.16%|-6.09%|-2.03%|-8.04%|-8.03%|-0.66%|-4.76%|-6.39%|-8.51%|-8.71%|
||Class C|-9.69%|+3.40%|-5.85%|-1.56%|-7.92%|-8.27%|-0.46%|-4.78%|-5.21%|-8.15%|-8.72%|
||Class D|-9.80%|+0.72%|-6.19%|-4.74%|-7.95%|-8.96%|-1.56%|-4.77%|-6.48%|-7.90%|-8.92%|
||Class E|-13.54%|+28.35%|-14.54%|-1.00%|-15.21%|-16.21%|+7.44%|-12.32%|-11.59%|-14.45%|-16.30%|
||Average|**-9.87%**|+8.33%|-7.42%|-1.95%|-9.00%|**-9.62%**|+0.71%|-6.09%|-7.02%|-9.27%|**-10.12%**|

<br>

### 3、Compared with the direct model
An accurate model is crucial for solving the over-filtering problem. However, the RDO-based and the skipping methods both adopt the inaccurate direct CNN model which is trained without considering the complex reference correlations across inter frames. To this end, the coding efficiency of the two methods can be further improved if a more accurate model is used.

+ Integrate our progressive model to the RDO-based and the skipping methods.

#### <p align="center">Table 4: BD-rate (%) of using the progressive model instead of the direct model in the RDO-based and skipping methods</p>

|Class|Sequence|LDP||RA||
|:---:|:---------:|:----:|:----:|:----:|:----:|
|||frame skipping|CTU-RDO|CU skipping|CTU-RDO|
|A|PeopleOnStreet|-4.21%|-7.10%|-7.63%|-8.16%|
||Traffic|-8.81%|-9.27%|-10.51%|-10.71%|
|B|BasketballDrive|-5.73%|-10.77%|-7.56%|-8.40%|
||BQTerrace|-9.38%|-12.73%|-10.46%|-11.91%|  
||Cactus|-8.42%|-10.13%|-9.23%|-10.30%|  
||Kimono|-4.72%|-7.34%|-5.75%|-5.93%|
||ParkScene|-4.03%|-5.23%|-6.84%|-6.99%|  
|C|BQMall|-8.18%|-11.00%|-9.98%|-10.42%|  
||PartyScene|-3.91%|-6.46%|-5.79%|-6.28%|  
||BasketballDrill|-9.76%|-12.60%|-11.00%|-11.93%|  
||RaceHorsesC|-2.84%|-5.43%|-5.22%|-5.85%|  
|D|BasketballPass|-8.03%|-11.15%|-9.08%|-10.02%|  
||BlowingBubbles|-4.19%|-6.93%|-6.63%|-7.22%|  
||BQSquare|-7.98%|-11.23%|-8.46%|-9.38%|  
||RaceHorses|-4.62%|-8.77%|-7.64%|-8.45%|  
|E|Johnny|-17.75%|-17.86%|-16.00%|-15.36%|  
||FourPeople|-16.08%|15.49%|-15.68%|-15.37%|  
||KristenAndSara|15.86%|-15.19%|-14.90%|-14.42%|  
|Average (progressive model)||-8.03%|-10.26%|-9.35%|-9.84%|  
|Average (direct model)||-7.42%|-9.00%|-7.02%|-9.27%|




<br>

### 4、Generalizability of our approach

+ Deploy our proposed scheme on different networks. In addition to VDCNN 23, the proposed approach is also implemented using existing networks for verification. Two networks, DSCNN and SEFCNN, are trained using the progressive method and the frame-level RDO is conducted for model selection.
	> From the results in Fig. 2 we can see that the direct model leads to over-filtering and the results are even worse than the H.265/HEVC anchor. The coding efficiency is improved with CTU-RDO.Furthermore, our proposed approach achieves comparable PSNR to that of CTU-RDO while the bitrate cost is reduced significantly

![](https://github.com/IVC-Projects/progressive_cnn_filter/blob/master/pictures/picture3.png)

	`Figure 2: The proposed approach is implemented using DSCNN and SEFCNN for comparison. In this example, the obtained progressive models are used to test the performance of inter frames in RA confifiguration at QP = 37.`

+ Deploy our proposed scheme on different configurations. In  addition, we apply our proposed scheme to another LDP configuration “IPPPIPPP”. 
	>From Table 6 we can see that the skipping  method achieves +0.35 dB PSNR improvement and -0.36% bitrate reduction over the anchor H.265/HEVC encoder. Using CTU-RDO, the bitrate reduction is slightly decreased and the PSNR performance is further boosted by +0.20 dB over the skipping method.In our proposed progressive scheme, the PSNR gain is the same as  that of CTU-RDO whereas the bitrate is decreased by -0.67%.

	#### <p align="center">Table 6: Performance under IPPPIPPP coding in LDP confifiguration</p>
	|Method|Anchor||frame skipping||CTU-RDO||Proposed||  
	|:---:|:---------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
	|||Bitrate|PSNR|∆Bitrate|∆PSNR|∆Bitrate|∆PSNR|∆Bitrate|∆PSNR|
	|Class A|6579.51|32.79|+0.06%|+0.31|-10.07%|+0.52|-0.31%|+0.52|  
	|Class B|3672.09|33.16|-0.51%|+0.22|-0.44%|+0.38|-0.76%|+0.36|  
	|Class C|1786.30|30.16|-0.57%|+0.36|-0.60%|+0.52|-0.77%|+0.52|  
	|Class D|509.57|29.88|-0.39%|+0.44|-0.47%|+0.58|-0.76%|+0.59|  
	|Class E|1448.55|36.22|-0.64%|+0.48|-0.65%|+0.87|-1.19%|+0.88|  
	|Average|2502.70|32.23|-0.36%|+0.35|-0.38%|+0.55|-0.67%|+0.55|


### 5、Visual quality.
Examples of filtered frames, such as the 13th  frames of sequence  “BQmall” and the 22th  frame of sequence “FourPeople”, processe  by the traditional in-loop filter, the direct CNN model, and our proposed progressive model.
>our progressive model successfully removes artifacts and retains some details. The results look visually more appealing.

![](https://github.com/IVC-Projects/progressive_cnn_filter/blob/master/pictures/picture2.png)

`Figure 3: Visual quality comparison of difffferent in-loop fifiltering schemes for QP = 37.`

