# 12/1 AI & Applications assignment page

## Title of out project

---

### *Memento - The Technical Solution for ‘Youngzheimer’*

## Members

---

| 이석철 (SUK CHEOL LEE) | Dept of. Information System | tjrcjf9@gmail.com |
| --- | --- | --- |
| 이하늘 (HA NUEL LEE) | Dept of. Information System | ezzeezz@naver.com |
| 정세희 (SE HEE JEONG) | Dept of. Information System | sjsk04230000@hanyang.ac.kr |
| 조준상 (JUN SANG CHO) | Dept of. Information System | cjsjustin99@naver.com |

## **I. Introduction**

---

치매는 대한민국 뿐만 아니라, 전세계적으로 환자가 꾸준히 증가하고 있는 추세이며 이에 따른 심각성 또한 증가하고 있습니다. 세계보건기구(WHO)에 따르면, 전 세계 치매인구는 약 5,000만명이며 2050년에는 이의 3를 넘는 1억 5,200만명이 될 것으로 추정됩니다. 치매는 아직까지 마땅한 치료 방법이 없으며, 건강한 삶을 살 수 있도록 다양한 활동을 하는 것이 치매 예방의 유일한 방법으로 알려져 있습니다. 치매의 위험군은 65세 이상의 고령층으로 알려져 있지만, 최근에는 비교적 젊은 층에 해당하는 40~50대에서 치매 환자가 늘어나는 추세입니다. 

이른바 '젊은 치매'로 불리는 '초로기치매'이며, 이로 인해 젊음을 뜻하는 young과 가장 대표적인 치매의 유형인 Alzheimer의 합성어인 **영츠하이머(Youngzheimer)**라는 신조어가 탄생하였습니다. 젊은 치매의 진행 속도는 65세 이상의 노인성치매보다 훨씬 빠르고 위험하지만, 정작 이를 해결하거나 예방할 수 있는 수단은 매우 부족합니다. 대한민국 정부는 치매 선별검사 비용을 지원하지만 이는 60세 이상의 노인에게만 해당되는 이야기이며, 시중에 존재하는 치매 예방 및 도우미 어플은 노인들을 겨냥해 만들어졌으며 그마저도 잦은 오류와 부족한 최적화로 혹평을 받는 실정입니다. 한국인의 사망원인 Top 10 안에 알츠하이머성 치매가 포함됨에도 정작 이를 예방하고 도움을 줄 수 있는 확실한 어플은 없는 수준입니다.  

이에 저희는 치매 예방에는 글을 읽거나 쓰는 등, 창조성을 요구하는 뇌 활동이 가장 효과적이기에, 시중의 어플과는 차이점을 주기 위해 **MEMENTO** 어플리케이션을 개발하였고, 본인의 하루 일과에서 있었던 일들을 토대로 사용자가 퀴즈를 진행하며 자연스레 하루를 되돌아볼 수 있도록 할 것입니다.

더불어, 치매 예방에 도움이 되는 좋은 습관을 젊은 세대가 조기에 만들어, 추후에 치매 발병률을 낮추고 국민 건강에 기여하는 것이 우리의 최종 목표입니다.

## **II. Datasets**

---

- **일상 타임라인 기반 Datasets 직접 제작 (약, 1500개)**
    - Reason 1 : 사용자가 시간대별로 무슨 일이 있었는지 가볍게 이야기 하는 일상 타임라인에 대한 데이터가 절대적으로 부족함
    - Reason 2 : 저작권 문제를 해결할 수 있는 충분한 타임라인 Dataset이 존재하지 않음
- **전처리(Preprocessing)**
    1. 불필요한 개행 문자 제거
    2. 답변 내 좌우 공백 제거
    3. Tokenizer 적용 후 데이터 수정
    4. 글자 수 6개 이하 제거
    5. 이상치 제거
- **전체 Datasets에서 Haystack annotationn Tool을 사용하여 질문지와 답변에 대한 Q&A Labeling 데이터 제작 (약, 100개)**
    
    **[육하원칙에 입각한 Q&A]**
    
    예시)
    
    **[when]** 석철이랑 제육볶음 먹었다 맛있었다를 어떤 날짜에 말했나요 ?
    
    **[when]** 석철이랑 제육볶음 먹었다 맛있었다를 말한 시간은 몇 시 인가요 ?
    
    **[where]** 석철이랑 제육볶음 먹었다 맛있었다를 말한 장소는 어디인가요 ?
    
    **[what]** 2022년 6월 26일 01시 43분에 정통집에서 무엇을 했나요 ?
    
    **[what] [감정기반]** 2022년 6월 26일 01시 43분에 정통집에서 어떤 감정을 느꼈나요 ?
    
    **[why] [감정기반] [꼬리질문]** 2022년 6월 26일 01시 43분에 왜 이런 감정을 느꼈나요 ?
    
    → 답변에 대한 정답은 No answer로, 모든 답변에 대해 정답처리 진행, 추후 본 대답을 통해 감정분석 진행 예정
    
    **[what]** 2022년 6월 26일 01시 43분에 정통집에서 한 말은 무엇인가요?
    
    **[who]** 2022년 6월 26일 01시 43분에 정통집에서 누구와 함께 행동했나요 ?
    
    → 혼자일 수도 있기 때문에 No Question/No answer일 수 있음
    
- **Question & Answer Labeling 데이터를 통한 데이터셋 Augmentation (약, 1000개 작업중)**
    1. (기존 Q&A 라벨링된 데이터를 활용하여) Question이 뽑히는 규칙 확인
        1. 학습 데이터 셋 비율 (Train : 80% / Val : 10% / Test : 10%)
        2. Question이 뽑히는 규칙 학습 (RoBERTa 모델) 
        3. Question Extenstion → Data Augmentation
    2.  (기존 Q&A 라벨링된 데이터를 활용하여) Answer이 뽑히는 규칙 확인
        1. 학습 데이터 셋 비율 (Train : 80% / Val : 10% / Test : 10%)
        2. Answer이 뽑히는 규칙 학습 (RoBERTa 모델)
        3. Answer Extenstion → Data Augmentation
- **Pretrained Model 제작**
    - KorQuAD 1.0으로 완성된 Q&A 데이터에 Pretrain
    - KorQuAD : 한글의 단어, 문장 간 유사도 계산, 다른 모형의 입력으로 사용가능하도록 하는 한글 기계독해를 위한 Datasets

**✓ 데이터 셋 구축 개수 : 110개의 타임라인 데이터를 六何原則(육하원칙)별로 나눈  651개의 지문**

**✓ 데이터 셋 개수 : 지문(Context)-질문(Question)-답변(Answer)형식의 MRC(주어진 상황에서 질의응답함) 데이터 (약, 1000개, 진행중)**

## **III. Methodology**

---

### [**모델 구조]**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b0864276-a42d-4f7a-99c4-caf9d37cfbab/Untitled.png)

- **CDQA(Closed Domain Qustion Answering: 특정 Domain의 Database에서 Q&A를 수행하는 Task)**
    1. 사용자가 녹음한 타임라인을 INPUT으로 받음
    2. Retriever와 Reader Model을 거쳐 질문과 대답을 반환
    
- **Retriever Model**
    - Elasticsearch
        - 문장 안의 키워드를 인덱스로 만들어 빠른 검색 가능
        - 내용 전체를 인덱싱해 특정 단어가 포함된 Content 전문 검색 가능
        - 시간복잡도(O) 비교
            - 기존 RDB(관계형 데이터베이스)의 시간복잡도 : O(n)
            - Elasticsearch의 시간복잡도 : O(1)
        - 질문 Q와 Content간 유사도 계산
            - TF-IDF를 통한 계산
            
            ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/31654600-3a31-4f30-a1b1-afe3de3f1f30/Untitled.png)
            
            - TF(Term Frequency) : 특정 문서에서 특정 단어의 등장 빈도
            - IDF(Inverse Document Frequency) : 전체 문서에서 특정 단어의 등장 빈도의 역수
            - BM25 : TF-IDF의 parameter를 변경하여 Best Match를 찾음
- Reader Model
    - RoBERTa-Large(MRC SOTA) model
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3527def5-0db2-444b-a4c6-4af310b6f7a2/Untitled.png)
    
    - QA Task에 최적화 시키기 위해 QA 데이터 셋을 **KorQuAD 1.0** 으로 Pretrain 학습시킴(한글 Corpus에 대한 Embedding)
        
        ❕→ KorQuAD + Our Data
        
    - Domain에 최적화 시키기 위해 RoBERTa-large를 활용하여 Our Datasets으로 Fine Tuning 학습시킴(분석에 맞게 업데이트)
        
        ❕→ RoBERTa-large (KorQuAD + Our Data)
        

## **IV. Evaluation & Analysis**

---

- **Retriever model 평가 지표**
    
    Retrieval accuracy : retriever의 성능 측정을 위한 metric
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ec031c16-a504-413b-8054-2711b9d526bb/Untitled.png)
    
- **최종 Retriever model의 Retrieval accuracy**
    
    → Accuracy와 Turn-Around Time(사용자가 타임라인을 녹음하고, 결과를 내기까지 걸리는 시간)을 고려해야 함
    
    → 진행중
    
- **질문 유형별 분포 확인**
    
    →진행중
    
- **두 번의 Fine-tuning 실험 성능평가 결과**
    - RoBERTa-large(KorQuAD + Out Data)의 F1 Score 확인
    - Exact Match (EM): 모델의 예측과, 실제 답이 정확하게 일치할 때만 점수 부여
        - 예측이 정세희 이고, 답이 정세희 언니일 때 EM Score는 0점
    - F1 score: Precision과 Recall을 계산하여 부분 점수를 제공
        - 예측이 정세희 답이 정세희 언니일 때 F1 score 는 precision이 1, recall이 0.5이므로 0.67
    
    → F1 Score을 주요 평가지표로 선정
    
    → 진행중
    

## **V. Related Work (Ex. existing studies)**

---

- **Tools, libraries, blogs, or any documentation that you have used to do this project.**
- **Tools**
    1. Jupyter notebook
    2. Google colaboratory
    3. Docker
    4. Haystack(Q&A annotation Tool)
    5. Tensorflow

- **Documentation**
    1. Fine-tuning Strategies for Domain Specific Question Answering under Low Annotation Budget Constraints
        
        [Fine-tuning Strategies for Domain Specific Question Answering under Low Annotation Budget Constraints](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7a5cc29d-60dc-41e2-a0c8-1f2d73818cc5/363_fine_tuning_strategies_for_dom.pdf)
        
    2. EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks
        
        [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/39804328-6f7c-4ff2-a092-4246780dfa6b/1901.11196.pdf)
        
    3. AEDA: An Easier Data Augmentation Technique for Text Classification
        
        [AEDA: An Easier Data Augmentation Technique for Text Classification](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7fb5a8d5-41e2-422c-8caa-862104e91bcb/2108.13230.pdf)
        

## **VI. Conclusion: Discussion**

---

❗**타임라인을 활용한 QA모델 생성의 의의**

- **하루를 기억하기에 어려움이 있던 사용자를 위한 영츠하이머 예방 QA 모델 생성**
- **STT(Speach to Text)서비스에서 생성된 데이터를 활용할 수 있는 새로운 방향 제시**

❗**새로운 데이터셋 제시**

- **Closed Domain QA모델 생성에 도움을 줌 ( Open Domain QA X)**

❗**추후 발전 방향** 

- **추가 데이터 수집**
- **잘 맞추지 못하는 질문 유형들에 대한 성능 개선**
- **일반화된 데이터를 통한 학습**
- **질문 유형의 다양화**
- **답변에 따른 감성분석 예정**
