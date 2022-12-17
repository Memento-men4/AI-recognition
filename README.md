Tech blog link : https://url.kr/qhn9ab



## **I. Datasets**

---

- **일상 타임라인 기반 Datasets 직접 제작 (약, 6621개)**
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
    
- **Question & Answer Labeling 데이터를 통한 데이터셋 Augmentation**
    1. (기존 Q&A 라벨링된 데이터를 활용하여) Question이 뽑히는 규칙 확인
        1. 학습 데이터 셋 비율 (Train : 90% / Test : 10%)
        2. Question이 뽑히는 규칙 학습 (RoBERTa 모델) 
        3. Question Extenstion → Data Augmentation
    2.  (기존 Q&A 라벨링된 데이터를 활용하여) Answer이 뽑히는 규칙 확인
        1. 학습 데이터 셋 비율 (Train : 90% / Test : 10%)
        2. Answer이 뽑히는 규칙 학습 (RoBERTa 모델)
        3. Answer Extenstion → Data Augmentation
        
- **Pytorch Lightning Dataset 활용**
    
    답변 인식 질문 생성을 원하지 않을 때 특정 `sep` token을 사용하여 예측하려는 부분을 분리하고 `MASK` token을 통해 답변을 대신 전달 하고자 함
    
- **Pretrained Model 제작**
    - KorQuAD 1.0으로 완성된 Q&A 데이터에 Pretrain
    - KorQuAD : 한글의 단어, 문장 간 유사도 계산, 다른 모형의 입력으로 사용가능하도록 하는 한글 기계독해를 위한 Datasets

**✓ 데이터 셋 구축 개수 : 110개의 타임라인 데이터를 六何原則(육하원칙)별로 나눈  651개의 지문**

**✓ 데이터 셋 개수 : 지문(Context)-질문(Question)-답변(Answer)형식의 MRC(주어진 상황에서 질의응답 함) 데이터**

- KorQuAD Data : Overall_data : 70,079개, Train_data : 60,407개, Dev_data : 5,774개, Test_data : 3,898개
- Our Data : Overall_data : 6621개, Train_data : 5959개, Test_data : 670개
- Our LG Data : Overall_data : 648개


## **II. Methodology**

---

### [**모델 구조]**

- **CDQA(Closed Domain Qustion Answering: 특정 Domain의 Database에서 Q&A를 수행하는 Task)**
    1. 사용자가 녹음한 타임라인 음성 text를 INPUT으로 받음
    2. Retriever와 Reader Model을 거쳐 질문과 대답을 즉각 생성
    3. DATABASE에 하루동안의 질문과 대답을 저장
    4. 퀴즈 요청 event가 발생하면 사용자에게 DATABASE에 저장된 퀴즈를 AI SPEAKER을 통해 출제 
    
- **Retriever Model**
    - Elasticsearch
        - 문장 안의 키워드를 인덱스로 만들어 빠른 검색 가능
        - 내용 전체를 인덱싱해 특정 단어가 포함된 Content 전문 검색 가능
        - 시간복잡도(O) 비교
            - 기존 RDB(관계형 데이터베이스)의 시간복잡도 : O(n)
            - Elasticsearch의 시간복잡도 : O(1)
        - 질문 Q와 Content간 유사도 계산
            - TF-IDF를 통한 계산

            - TF(Term Frequency) : 특정 문서에서 특정 단어의 등장 빈도
            - IDF(Inverse Document Frequency) : 전체 문서에서 특정 단어의 등장 빈도의 역수
            - BM25 : TF-IDF의 parameter를 변경하여 Best Match를 찾음

- **Reader Model**
    - RoBERTa-Large(MRC SOTA) model
    
    - QA Task에 최적화 시키기 위해 QA 데이터 셋을 **KorQuAD 1.0** 으로 Pretrained 학습시킴(한글 Corpus에 대한 Embedding)
        
        ❕→ KorQuAD + Our Data
        
    - Domain에 최적화 시키기 위해 RoBERTa-large를 활용하여 Our Datasets으로 Fine Tuning 학습시킴(분석에 맞게 업데이트)
        
        ❕→ RoBERTa-large (KorQuAD + Our Data)
        

## **III. Evaluation & Analysis**

---

- **Our** **Model** **Evaluation**
    
- **두 번의 Fine-tuning 실험 성능평가 결과**
    
    - RoBERTa-large(KorQuAD + Out Data)의 F1 Score 확인
    - Exact Match (EM): 모델의 예측과, 실제 답이 정확하게 일치할 때만 점수 부여
        - ex. 예측이 정세희 이고, 답이 정세희 언니일 때 EM Score는 0점
    - F1 score: Precision과 Recall을 계산하여 부분 점수를 제공
        - 실제 정답과 예측치의 겹치는 부분을 고려한 점수로, EM보다 완화된 평가 척도
        - 영문과 달리 한국어에는, 어절(띄어쓰기 단위) 내 다양한 형태소 활용을 위해 점수가 다소 낮게 측정되므로, 음절(글자) 단위의 F1 Score를 사용
        - ex. 예측이 정세희 답이 정세희 언니일 때 F1 score 는 precision이 1, recall이 0.5이므로 0.67
    
    → F1 Score을 주요 평가지표로 선정
    
    [최종 모델의 평가 결과]
    
    | F1 | EM |
    | --- | --- |
    | 79.7866 | 64.2857 |

## **IV. Related Work (Ex. existing studies)**

---

- **Tools**
    1. Jupyter notebook
    2. Google colaboratory
    3. Docker
    4. Haystack(Q&A annotation Tool)
    5. Tensorflow
    6. Pytorch

- **Library**
    
    
    | pandas | numpy | torch | transformers | tokenizers |
    | QuestionAnsweringTrainer | postprocess_qa_predictions | torchtext | Elasticsearch | pytorch-lightning|
    | torchmetrics | tokenizer | List | Dict | tqdm.notebook |
    | json | Path | Dataset | DataLoader | re |
    | os | random | Counter | string | argparse |
    | sys | glob | logging | timeit | eval_during_train |
    | Optional | WEIGHTS_NAME | AdamW | AlbertForQuestionAnswering | AlbertTokenizer |
    | BertConfig | BertForQuestionAnswering | BertTokenizer | DistilBertConfig | DistilBertForQuestionAnswering |
    | RobertaConfig | RobertaForQuestionAnswering | RobertaTokenizer | XLMConfig | XLMForQuestionAnswering |
    | XLMTokenizer | XLNetConfig | XLNetForQuestionAnswering | XLNetTokenizer | get_linear_schedule_with_warmup |
    | squad_convert_examples_to_features | SquadResult | SquadV1Processor | SquadV2Processor | KoBertTokenizer |
    | PredictionOutput | Trainer | is_torch_tpu_available | wandb_mixin | tensorflow |
- **Documentation**
    1. Fine-tuning Strategies for Domain Specific Question Answering under Low Annotation Budget Constraints
        
        [Fine-tuning Strategies for Domain Specific Question Answering under Low Annotation Budget Constraints](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7a5cc29d-60dc-41e2-a0c8-1f2d73818cc5/363_fine_tuning_strategies_for_dom.pdf)
        
    2. EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks
        
        [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/39804328-6f7c-4ff2-a092-4246780dfa6b/1901.11196.pdf)
        
    3. AEDA: An Easier Data Augmentation Technique for Text Classification
        
        [AEDA: An Easier Data Augmentation Technique for Text Classification](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7fb5a8d5-41e2-422c-8caa-862104e91bcb/2108.13230.pdf)
        
    4. RoBERTa: A Robustly Optimized BERT Pretraining Approach
        
        [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/609a64ef-f9b1-4914-a040-5327a3be0d7b/1907.11692.pdf)
        

## **V. Conclusion: Discussion**

---

❗**타임라인을 활용한 QA모델 생성의 의의**

- **하루를 기억하기에 어려움이 있던 사용자를 위한 영츠하이머 예방 QA 모델 생성**
- **STT(Speach to Text)서비스에서 생성된 데이터를 활용할 수 있는 새로운 방향 제시**

❗**새로운 데이터셋 제시**

- **Closed Domain QA모델 생성에 도움을 줌 ( Open Domain QA X )**

❗**추후 발전 방향** 

- **추가 데이터 수집**
- **잘 맞추지 못하는 질문 유형들에 대한 성능 개선**
- **일반화된 데이터를 통한 학습**
- **질문 유형의 다양화**
- **답변에 따른 감성분석 예정**
- **답변 결과에 따른 질문 사용자화**
