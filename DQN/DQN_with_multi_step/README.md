# スクリプト概要　　
* Multi Step Learningを使用したDQNの学習・実行機能を実装したスクリプト  

# 実行方法
## 学習
* 下記コマンドを実行  
`> python ../main.py -a DQN_with_multi_step -e <環境名>`　　
- 下記オプションを指定できる  
  -e : ゲーム環境の種類（'CartPole', 'Catcher', 'Mario'）  

* 学習が完了すると、学習したエージェントで1エピソード実行される  
* 実行するとlogフォルダ以下にタイムスタンプ名のフォルダが作成され、ログと学習結果が保存される  

## 実行  
* 下記コマンドを実行  
`> python ../main.py -a DQN_with_multi_step e <環境名> -p <学習済みモデルのパス> --play`　　
- 下記オプションを指定できる  
  -e : ゲーム環境の種類（'CartPole', 'Catcher', 'Mario'）  
  -p : 学習済みモデルのパス  

* 学習済みモデルのパスは「学習コマンド」実行時に生成されたログフォルダに保存される.ptファイルを指定する  
* 環境名は学習時に指定した環境と同じものを指定する

---
# Multi Step Learningの概要  
## Multi Step Learningとは  
* nステップ分の報酬と状態価値からTD誤差を計算する  
* Atariのゲームではn=3が良いとされている  

    例）n=3の場合  
<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+r_%7Bt%2B1%7D%2B%5Cgamma+r_%7Bt%2B2%7D+%2B+%5Cgamma%5E2+r_%7Bt%2B3%7D+%2B+%5Cgamma%5E3+%5Cunderset%7Ba%7D%7Bmax%7D+Q%28s_%7Bt%2B3%7D%2C+a%29+-+Q%28s_t%2C+a_t%29" 
alt="r_{t+1}+\gamma r_{t+2} + \gamma^2 r_{t+3} + \gamma^3 \underset{a}{max} Q(s_{t+3}, a) - Q(s_t, a_t)">  

---
# 実装メモ  
* キューを使用して実現している  
<img src="../../docs/DQN/DQN_with_multi_step/multi_step.jpg">  