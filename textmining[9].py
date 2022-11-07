# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import glob
from afinn import Afinn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import numpy as np
import matplotlib.pyplot as plt

# !pip install afinn

pos_review= (glob.glob("C:/Users/student/Downloads/데이터/aclImdb/train/pos/*.txt"))[20]

f = open(pos_review, 'r')
lines1 = f.readlines()[0]
f.close()

afinn= Afinn()

afinn.score(lines1)

neg_review= (glob.glob("C:/Users/student/Downloads/데이터/aclImdb/train/neg/*.txt"))[20]

f = open(neg_review, 'r')
lines2 = f.readlines()[0]
f.close()

afinn.score(lines2)

NRC = pd.read_csv('C:/Users/student/Downloads/데이터/nrc.txt', engine = "python",header= None,sep= "\t")

NRC = NRC[(NRC != 0).all(1)]

NRC = NRC.reset_index(drop = True) # 감성어와감성표현이유의미한라벨들만추출

tokenizer = RegexpTokenizer('[\w]+')

stop_words= stopwords.words('english')

p_stemmer= PorterStemmer()

raw = lines1.lower()
tokens = tokenizer.tokenize(raw)
stopped_tokens= [i for i in tokens if not i in stop_words]

match_words= [x for x in stopped_tokens if x in list(NRC[0])]

emotion =[]
for i in match_words:
    temp = list(NRC.iloc[np.where(NRC[0] == i)[0], 1])
    for j in temp:
        emotion.append(j)

sentiment_result1 = pd.Series(emotion).value_counts()

sentiment_result1

sentiment_result1.plot.bar()

raw = lines2.lower()
tokens = tokenizer.tokenize(raw)
stopped_tokens= [i for i in tokens if not i in stop_words]

match_words= [x for x in stopped_tokens if x in list(NRC[0])]

emotion = []
for i in match_words:
    temp = list(NRC.iloc[np.where(NRC[0] == i)[0], 1])
    for j in temp:
        emotion.append(j)

sentiment_result2 = pd.Series(emotion).value_counts()

sentiment_result2

sentiment_result2.plot.bar()

# +
#지도학습 기반 감성분석
#영화 리뷰(IMDB) 감성을 분석

import pandas as pd
import glob
from afinn import Afinn
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# -

#긍정 리뷰 학습 집합 불러오기
pos_review = (glob.glob("C:/Users/student/Downloads/데이터/aclImdb/train/pos/*.txt"))

lines_pos=[]
for i in pos_review:
    try:
        f = open(i,'r')
        temp = f.readlines()[0]
        lines_pos.append(temp)
        f.close()
    except Exception as e:
        continue

len(lines_pos)

#부정 리뷰 학습 집합 불러오기
neg_review=(glob.glob("C:/Users/student/Downloads/데이터/aclImdb/train/neg/*.txt"))

lines_neg=[]
for i in neg_review:
    try:
        f = open(i,'r')
        temp = f.readlines()[0]
        lines_neg.append(temp)
        f.close()
    except Exception as e:
        continue

len(lines_neg)

#긍정과 부정리뷰 학습 집합 결합하기
total_text = lines_pos + lines_neg

len(total_text)

#긍정과 부정 클래스를 라벨링하기
x = np.array(["pos","neg"])
class_Index = np.repeat(x,[len(lines_pos),len(lines_neg)],axis=0)

stop_words = stopwords.words('english')

#단어들의 가중치를 부여하고, 문서 - 단어 매트릭스로 변환하기
vect = TfidfVectorizer(stop_words = stop_words).fit(total_text)

X_train_vectorized = vect.transform(total_text)

X_train_vectorized.index = class_Index

#로지스틱 회귀 모델 구성하기
from sklearn.linear_model import LogisticRegression, SGDClassifier
model = LogisticRegression()
model.fit(X_train_vectorized, class_Index)

#긍정 리뷰 평가 집합의 리뷰 평가하기
pos_review_test = (glob.glob("C:/Users/student/Downloads/데이터/aclImdb/test/pos/*.txt"))[10]

test=[]
f = open(pos_review_test,'r')
test.append(f.readlines()[0])
f.close()

predictions=model.predict(vect.transform(test))

predictions

neg_review_test = (glob.glob("C:/Users/student/Downloads/데이터/aclImdb/test/neg/*.txt"))[20]

test2 = []
f = open(neg_review_test, 'r')
test2.append(f.readlines()[0])
f.close()

predictions = model.predict(vect.transform(test2))

predictions

#의사결정 나무 모델 구성하기
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train_vectorized, class_Index)

predictinos = clf.predict(vect.transform(test))

predictions

predictions = clf.predict(vect.transform(test2))

predictions

#서포트 벡터 머신 모델 구성하기
from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train_vectorized, class_Index)

predictions=clf.predict(vect.transform(test))

predictions

predictions=clf.predict(vect.transform(test2))

predictions
