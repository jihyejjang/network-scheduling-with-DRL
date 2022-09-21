# DDQN을 활용한 강화학습 기반 타임슬롯 스케줄링(**Timeslot scheduling using Double Deep Q-Network**)

- 대학원 랩실에서 진행한 개인 연구
- 딥러닝을 사용한 강화학습 알고리즘인 DDQN(Double Deep Q-Network)를 활용하여, 네트워크 패킷스케줄링 문제 해결방법 연구
    - simpy를 사용해 네트워크 시뮬레이션 구현 (packet generation, link, node, packet transmission ,...)
    - tensorflow로 DDQN을 구현해, 네트워크 상황에 따른 scheduling action을 최적화
    - 기존 스케줄링 알고리즘보다 높은 성능을 보임으로서, 강화학습을 네트워크에 도입할 수 있는 가능성을 보임

## Description


- 2022년 7월호 ETRI 게재
- IEEE access revision 진행중

### 🔎 Abstract

증폭되는 네트워크 트래픽 양에 따라 엄격해지는 requirement를 충족하기 위해 딥러닝을 잠재적인 솔루션으로 논의되는 단계에서, MDP로 모델링되기에 좋은 네트워크에 강화학습을 적용하는 연구가 점점 많이 이루어지고 있음. 이에 따라 본 연구에서는 타임슬롯 기반 네트워크 환경을 Simpy로 구현하고 DDQN으로 학습하여 단순히 학습환경에서 뿐만 아니라 확장된 네트워크 환경과 다양한 시나리오에 적응할 수 있음을 보여줌

### 🎙 Summary

- DDQN 을 활용하여, network 상황에 따라 priority queue에서의 Serve를 조절함으로서 packet scheduling 학습


- 비교를 위한 기존 휴리스틱 알고리즘
    - SP : 하위 우선순위의 전송이 보장 되지 않음
    - WRR : network의 utilization에 따른 weight조절이 필요함
- 연구 진행
    1. timeslot scheduling simulation with heuristic algorithms : 학습에 사용할 timeslot size, flow generation process, network parameter(deadline, generation period,…)를 고정하기 위해 기존 알고리즘으로 시뮬레이션하여 어느 수준의 성능을 보이는지 확인하고, 학습 결과와 비교하기 위한 지표 생성
    2. Training agent in a single node environment : output port가 1개인 single node에서 priority queue scheduling을 학습
      <img width="488" alt="node" src="https://user-images.githubusercontent.com/61912635/190987229-ee093194-1c38-43fa-b9db-0cc7d37b7ce6.png">           
    3. Test in network topology : 8개의 flow가 지정된 route로 전송되는 실제 네트워크와 비슷한 topology환경에서 학습된 DDQN agent를 적용하여 test simulation 진행  
      <img width="366" alt="topology" src="https://user-images.githubusercontent.com/61912635/190987220-1dba8aed-761e-4cf1-9be9-428c42b72cd8.png">

### ✔️ Result

score (accumulated reward of an episode)가 existing algorithms를 능가 ⇒ 각각 priority의 packet들을 deadline안에 전송하는 비율이 DDQN agent가 더 높음 

- learning curve (In a single node envorinment)  
<img width="488" src="https://user-images.githubusercontent.com/61912635/190987235-9dd59ac4-32d7-4a41-8c80-ac06dcd79dbd.png">  

- Loss    
<img width="488" src="https://user-images.githubusercontent.com/61912635/190987230-bd67261b-99bb-48d7-96a3-eb11e6d8d853.png">  
    
- test result in topology at each scenario  
<img width="488" src="https://user-images.githubusercontent.com/61912635/190987232-cd48bd9f-b6bb-470b-8d64-c750c6ab51f4.png"> 

### 👩🏻‍💻 Development
    

### 🛠 Source code

- ddqn
    - node.py : Packet receive/send, state observing, applying action selected
    - src.py: Flow configuration & generation
    - env.py : Network simulation & training (packet forwarding to destination according to its route by timeslot)
    - test.py : Test the model trained in env.py
    - ddqn.py , agent.py : Conducting DDQN algorithm where saving data <s,a,r,s’> in replay memory , training(model fitting),and action selection, etc,..




Paper
----

[류지혜, 권주혁 and 정진우. "DDQN을 활용한 강화학습 기반 타임슬롯 스케줄링" 한국통신학회논문지 47, no.7 (2022) : 944-952.doi: 10.7840/kics.2022.47.7.944](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002861752)

