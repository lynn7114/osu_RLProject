# RLProject
## 폴더별 역할 정리 

main.py: 프로그램 실행 시작점

beatmap.py: .osu 파일을 읽고 노트·타이밍 정보를 파싱

environment.py: gym.Env 클래스로 RL 환경 구현

train_dqn.py: DQN 학습시키는 스크립트(stable-baselines3 로 학습 (실험 코드))

models/: 학습된 모델이 저장됨

beatmaps/: 학습용 맵(.osu) 파일들

data/: 원본 .osz 자료