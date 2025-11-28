# 리듬게임 osu! 강화학습 프로그램 
## 폴더별 역할 정리 

main.py: 

+ 프로그램 실행 시작점
+ random number generator seed 변경을 통한 실험 및 신뢰구간 작성
+ 노래 선택 함수 불러온 후 알고리즘 선택하게 함
+ 알고리즘 함수 불러와서 실행

beatmap.py: .osu 파일을 읽고 노트·타이밍 정보를 파싱

environment.py: gym.Env 클래스로 RL 환경 구현

evaluate.py: 학습된 알고리즘들의 성능을 통합, 요약, 시각화하여 최종 비교 분석을 수행

song_manager.py: 

- data 폴더 내 .osz 파일들을 자동으로 번호 매겨 출력
- 사용자 입력으로 곡 선택
- 선택한 .osz 압축 해제 후 .osu 파일을 찾아 반환
- 난이도에 따라 나뉜 .osu 파일 중에 사용자 입력으로 곡 선택

models/: 학습된 모델이 저장됨

beatmaps/: 학습용 맵(.osu) 파일들

data/: 원본 .osz 자료

