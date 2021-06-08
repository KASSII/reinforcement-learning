# スクリプト概要　　
* DDQNの学習・実行機能を実装したスクリプト  

# 実行方法
## 学習
* 下記コマンドを実行  
`> python ../main.py -a DDQN -e <環境名>`　　
- 下記オプションを指定できる  
  -e : ゲーム環境の種類（'CartPole', 'Catcher', 'Mario'）  

* 学習が完了すると、学習したエージェントで1エピソード実行される  
* 実行するとlogフォルダ以下にタイムスタンプ名のフォルダが作成され、ログと学習結果が保存される  

## 実行  
* 下記コマンドを実行  
`> python ../main.py -a DDQN e <環境名> -p <学習済みモデルのパス> --play`　　
- 下記オプションを指定できる  
  -e : ゲーム環境の種類（'CartPole', 'Catcher'）  
  -p : 学習済みモデルのパス  

* 学習済みモデルのパスは「学習コマンド」実行時に生成されたログフォルダに保存される.ptファイルを指定する  
* 環境名は学習時に指定した環境と同じものを指定する

---
# DDQNの概要  
## DDQNとは  
* Fixed Target Q-Networkをより安定させた方法  
* 次状態s'での行動a'はメインネットワークから求め、Q(s', a')はターゲットネットワークから求める  

<u>DQNのTD誤差</u>  
<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+r%2B%5Cgamma+%5Cunderset%7Ba%27%7D%7Bmax%7DQ_t%28s%27%2C+a%27%29+-+Q_m%28s%2Ca%29%0A" 
alt="r+\gamma \underset{a'}{max}Q_t(s', a') - Q_m(s,a)
">

<u>DDQNのTD誤差</u>  
<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+a%27+%3D+%5Cunderset%7Ba%7D%7Bargmax%7D+Q_m%28s%27%2C+a%29%0A" 
alt="a' = \underset{a}{argmax} Q_m(s', a)
">

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+r%2B%5Cgamma+Q_t%28s%27%2C+a%27%29+-+Q_m%28s%2Ca%29%0A" 
alt="r+\gamma Q_t(s', a') - Q_m(s,a)
">
