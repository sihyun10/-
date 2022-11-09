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

pip install bs4

import requests as req
from bs4 import BeautifulSoup as bs

res = req.get("https://movie.naver.com/movie/bi/mi/pointWriteFormList.naver?code=157178&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false")
soup = bs(res.text,'lxml')

soup.select('div.score_reple > p')

viewer = soup.select('span.ico_viewer') #관람객 요소 선택

for i in viewer:
    i.extract()  #extract() : 추출하다

#'관람객' 요소가 삭제된 해당 요소를 선택하고 추출하기
review = soup.select('div.score_reple > p')
for i in review:
    print(i.text)

#개행(공백)문자 삭제하기
for i in review:
    print(i.text.strip())  #개행(공백) 문자 삭제

# +
#1~575페이지까지의 리뷰를 수집해보자!
#페이지 수가 넘어갈 때마다 URL 맨 뒤의 숫자도 바뀜


for i in range(1,575):
    res = req.get("https://movie.naver.com/movie/bi/mi/pointWriteFormList.naver?code=157178&type=after&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page="+str(i))
    soup = bs(res.text,"lxml")
    viewer=soup.select('span.ico_viewer')
    for i in viewer:
        i.extract()
    review = soup.select("div.score_reple > p")
    for i in review:
        print(i.text.strip())
# -


