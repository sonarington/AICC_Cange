# 1. 프로젝트 소개

#### 얼굴 분석 기반 헤어스타일 추천 및 퍼스널 컬러 예측 서비스
헤어, 패션, 화장에 있어서 더 나은 선택을 할 수 있도록 얼굴 분석을 통해 정보를 제공합니다. 이미지 데이터를 이용하여 얼굴형 분석 기반 헤어스타일을 추천하고, 피부색 분석 기반 퍼스널컬러를 예측합니다.
또한, 유저 간 채팅방을 만들어 더욱 다양한 정보 공유가 이루어질 수 있도록 하고 있습니다. 각 퍼스널 컬러 및 얼굴형 별 채팅방을 만들어 놓았고, 유저가 원하는 채팅방에 들어가 유저끼리 대화를 나눌 수 있습니다.

![376439365-f7e095dd-2e1b-4b8f-b1c0-812be43bbe6e](https://github.com/user-attachments/assets/afe43fd8-d292-42b3-87e9-5f8d54a98b8a)

allrange라는 로고는 모든 범위를 다 다뤄보겠다는 의미로 선정하였고, 오렌지와 비슷한 발음으로 대표 컬러 또한 오렌지색(#FF9A42)으로 선택하였습니다.


# 2. 프로젝트 구조
![376442376-8bc56d2a-aab8-40aa-9054-1fec2b7d1066](https://github.com/user-attachments/assets/5a2ccc7c-2f9d-4858-baf2-c4792edad52a)

# 3. 웹 페이지 사용 방법
#### (1) 사진 첨부 방법
먼저 성별을 선택하고 원하는 사진을 첨부합니다. 원하는 분석 방법(1. 좌표 이용, 2. 모델 이용)을 선택하여 분석 시작 버튼을 누른 후 잠시 기다리면 결과창으로 이동할 수 있는 버튼이 나타납니다. 얼굴형은 둥근형, 장방형으로 나누어지며 얼굴형 결과 페이지에서 각 얼굴형의 특징과 베스트 헤어, 워스트 헤어를 추천해드립니다. 퍼스널컬러 분석도 마찬가지이며, 퍼스널 컬러 결과 페이지에서는 잘 어울리는 색과 어울리지 않는 색을 안내하고 있습니다.
#### (2) 웹캠 이용 방법
웹캠을 이용하여 캡쳐 사진으로 분석하는 방법은 카메라 켜기 버튼을 누르고 성별을 선택 후 사진 찍기 버튼을 누르면 화면 캡쳐가 됩니다. 이 캡쳐된 사진으로 사진 첨부와 같은 방식으로 분석 버튼을 눌러 결과를 확인하실 수 있습니다.
#### (3) 채팅방 사용
채팅방은 회원/비회원으로 사용 가능합니다. 회원인 경우 로그인 후, 비회원인 경우 닉네임 입력 후 채팅 홈에 들어갈 수 있습니다. 퍼스널컬러와 얼굴형 별로 채팅방이 생성되어 있으니 원하는 유형의 채팅방에 들어가 자유롭게 정보 공유를 하며 유저 간 채팅을 나눌 수 있습니다.

# 4. 활용 장비 및 재료

(1) 개발 환경: VS Codel, Github
    
(2) 프레임워크 및 라이브러리:  Vite, React, Express, Flask, MySQL
    
(3) 기술 스택: JavaScrpit, Python, CSS
    
(4) 기타 툴: Figma


# 5. 팀원 소개

#### 박근만
```
email: spiritington@gmail.com
Github: https://github.com/sonarington
담당: 팀장 및 프로젝트 총괄, 채팅 웹 소켓 연결(프론트/백), 로그인 및 회원가입(백엔드), 서버 연결
```

#### 김진호
```
email: mec273k@naver.com
Github: 
담당: 퍼스널컬러 분석 모델 구축(백엔드)
```

#### 손소희
```
email: dsoheeb806@gmail.com
Github: https://github.com/noisoi
담당: 얼굴형 분석 모델 구축(백엔드), 페이지 디자인 및 퍼블리싱(프론트엔드), FAQ 자동생성 AI봇(백엔드)
```

#### 김제우
```
email: kdamin123@naver.com
Github: 
담당: 자료 조사 및 분석 결과 페이지 디자인(프론트엔드)
```
