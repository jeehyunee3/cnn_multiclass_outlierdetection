# Effective Classification Method of Hierarchical CNN for Multi-Class Outlier Detection


### 다중 클래스 이상치 탐지를 위한 계층 CNN의 효과적인 클래스 분할 방법

> 본 연구에서는 MVTec-AD 데이터셋을 사용하여 다중 클래스 이상치 탐지에 계층형 CNN이 효과적인지 여부를 조사하였으며, 계층적 군집화 알고리즘을 사용하여 새로운 클래스를 생성하고 이를 사용하여 계층형 CNN을 구성하였습니다.

`합성곱신경망(Convolutional Neural Network)` `군집화(Clustering)` `계층적 구조(Hierarchical Structure)` `이미지 분류(Image Classification)`  `이상치 탐지(Anomaly Detection)` `딥 러닝(Deep Learning)` `다중 일반 클래스(Multiple Normal Classes)`
<br/>

## Research Objective
**1. 단층적 CNN과 계층적 CNN을 통한 Supervised 이상치 탐지**

**2. 효과적인 클래스 분할 방식 탐구 
	  (K-means clustering vs GMM clustering vs Hierarchical clustering)**

<br/>

## Model Architecture

![enter image description here](https://user-images.githubusercontent.com/72274498/236993748-1163e161-5a75-49c2-97b9-a360e24ceef9.png)
- **단층적(Single)  CNN**: 물체의 종류와 이상치를 한 번에 30개의 상태로 나누어 분류하는 단편적 형태의 CNN 모델
- **계층적(Hierarchical) CNN**: 물체의 종류를 분류하는 CNN 모델과 이상치를 분류하는 CNN 모델 2개를 사용하는 방법

- **클래스 분할기반 계층적(Hierarchical) CNN**: 물체의 종류를 분류하는 CNN 모델과 그룹의 이상치를 분류하는 CNN 모델 2개를 사용하는 방법
   - **K-means clustering** : k-means clustering 알고리즘을 사용하여 이미지 grouping
   - **GMM clustering** :  gmm clustering 알고리즘을 사용하여 이미지 grouping
   - **Hierarchical clustering**  : hierachical clustering 알고리즘을 사용하여 이미지 grouping

<br/>

## Paper

[다중 클래스 이상치 탐지를 위한 계층 CNN의 효과적인 클래스 분할 방법 - 한국컴퓨터정보학회 학술발표논문집 - 한국컴퓨터정보학회 : 논문 - DBpia](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11140311)
- 한국컴퓨터정보학회 
- 2022년 한국컴퓨터정보학회 하계학술대회 논문집 제 30권 2호, pg. 81-84, 2022.07
- 김지현, 이세영, 김예림, 안서영, 박새롬 
- 성신여자대학교

<br/>

## DATA
본 연구에서는 **MVTec**에서 제공하는 **MVTec-AD (Anomaly Detection)** 데이터 셋을 이용하였습니다. 

[MVTec Anomaly Detection Dataset: MVTec Software](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

**[데이터 수]**

![enter image description here](https://user-images.githubusercontent.com/72274498/236992664-9974aff6-a9ae-459d-bb66-04b756f2726e.png)

<br/>

# Performances

### 1. Single CNN Model
|이미지 사이즈| 컨볼루션 레이어 수 | Accuracy | F1-Score | Parameters |
|---|---|---|---|---|
|64*64| 2 | 0.7495 | 0.554 | 7,396,190 |
|128*128 | 2 | 0.765 | 0.5215 | 31,513,438 |
|256*256 | 2 | 0.679 | 0.4998 | 130,079,582 |
|128*128 | 5 | 0.7691 | 0.5927 | 7,433,118 |
|**128*128** | **5** | **0.7715** | **0.5258** | **429,086** |
|128*128 | 7 | 0.7428 | 0.5758 | 216,222 |

### 2. Hierarchical CNN Model
|Cluster | Accuracy | Parameters |
|---|---|---|
|Not Clusted Hierarchical Model |0.7838|6,809,261|
|K-means by 3 groups | 0.7930 | 1,703,573 |
|K-means by 5 groups | 0.7799 | 2,554,521 |
|K-means by 7 groups | 0.7018 | 3,405,469 |
|GMM by 3 groups | 0.7891 | 1,703,573 |
|GMM by 6 groups | 0.7780 | 3,024,995 |
|**Hierarchical by 2 groups** | **0.7936** | **1,278,099** |
|Hierarchical by 5 groups | 0.7897 | 2,554,521 |
|Hierarchical by 7 groups | 0.7643 | 3,405,469 |

