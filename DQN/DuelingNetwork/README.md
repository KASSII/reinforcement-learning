# スクリプト概要　　
* Dueling Networkの学習・実行機能を実装したスクリプト  

# 実行方法
## 学習
* 下記コマンドを実行  
`> python ../main.py -a DuelingNetwork -e <環境名>`　　
- 下記オプションを指定できる  
  -e : ゲーム環境の種類（'CartPole', 'Catcher', 'Mario'）  

* 学習が完了すると、学習したエージェントで1エピソード実行される  
* 実行するとlogフォルダ以下にタイムスタンプ名のフォルダが作成され、ログと学習結果が保存される  

## 実行  
* 下記コマンドを実行  
`> python ../main.py -a DuelingNetwork e <環境名> -p <学習済みモデルのパス> --play`　　
- 下記オプションを指定できる  
  -e : ゲーム環境の種類（'CartPole', 'Catcher', 'Mario'）  
  -p : 学習済みモデルのパス  

* 学習済みモデルのパスは「学習コマンド」実行時に生成されたログフォルダに保存される.ptファイルを指定する  
* 環境名は学習時に指定した環境と同じものを指定する

---
# Dueling Networkの概要  
## Dueling Networkとは  
* Q値の出力方法を工夫したネットワークを使用するDQN派生のアルゴリズム  
  <b><u>DQN</u></b>  
  直接Q値を推定する  
  <b><u>Dueling Network</u></b>  
  状態価値V(s)、アドバンテージA(s,a)を推定する  

## アドバンテージについて  
* 行動価値は状態のみによって決まる部分と行動によって決まる部分に分解できる  
  状態のみによって決まる部分が状態価値、行動によって決まる部分を**アドバンテージ**という  
<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+Q%28s%2C+a%29+%3D+V%28s%29+%2B+A%28s%2C+a%29" 
alt="Q(s, a) = V(s) + A(s, a)">

  例）CartPole  
  倒れる直前は右に動こうが左に動こうが報酬は低くなる  
  → V(s)が大きく、A(s,a)が小さい状態ということ  

## ネットワークのイメージ  
<img src="../../docs/DQN/DuelingNetwork/DuelingNetwork.jpg">  

単純に出力を加算するだけだと、それぞれの出力がV(s)、A(s,a)を表現していると言えないので、工夫（制御)を加える必要がある  
(足して Q値になる組み合わせは無限にあるので、一意に定まるように制限を加える)  

ある方策πについて、状態価値V(s)と行動価値Q(s,a)は以下のように定義される  
<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+Q%5E%7B%5Cpi%7D%28s%2Ca%29+%3D+%5Cmathbb%7BE%7D%5BR_%7Bt%7D+%5Cmid+s_%7Bt%7D%3Ds%2C+a_%7Bt%7D%3Da%2C+%5Cpi%5D" 
alt="Q^{\pi}(s,a) = \mathbb{E}[R_{t} \mid s_{t}=s, a_{t}=a, \pi]">  
（行動価値は状態s、行動a、方策πの時の報酬和R_tの期待値）  
<br>

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+V%5E%7B%5Cpi%7D%28s%29+%3D+%5Cmathbb%7BE%7D_%7Ba%5C%7E%5Cpi+%28s%29%7D+%5BQ%5E%7B%5Cpi%7D%28s%2Ca%29%5D" 
alt="V^{\pi}(s) = \mathbb{E}_{a\~\pi (s)} [Q^{\pi}(s,a)]">  
（状態価値は全てのaに対するQ(s,a)の期待値）  
<br>

<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cpi%28s%29%3D%5Cunderset%7Ba%27%7D%7Bargmax%7DQ%28s%2Ca%27%29" 
alt="\pi(s)=\underset{a'}{argmax}Q(s,a')">
のような決定的な方策の場合、
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+V%28s%29%3DQ%28s%2C+a%27%29+" 
alt="V(s)=Q(s, a') ">

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+Q%28s%2C+a%29+%3D+V%28s%29+%2B+A%28s%2C+a%29" 
alt="Q(s, a) = V(s) + A(s, a)">
なので、
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+A%28s%2C+a%27%29+%3D+0" 
alt="A(s, a') = 0">
が成り立つ。  

上記のアドバンテージの条件を制約に加えると、行動価値は下記の式で計算できる  
<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+Q%28s%2Ca%29+%3D+V%28s%29%2B%28A%28s%2Ca%29-%5Cunderset%7Ba%27%5Cin+%7CA%7C%7D%7Bmax%7DA%28s%2C+a%27%29%29" 
alt="Q(s,a) = V(s)+(A(s,a)-\underset{a'\in |A|}{max}A(s, a'))">

定義とは少し異なるが、より安定する下記式を使う  
<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+Q%28s%2Ca%29+%3D+V%28s%29%2B%28A%28s%2Ca%29-%5Cfrac%7B1%7D%7B%7CA%7C%7D%5Csum_%7Ba%27%7DA%28s%2Ca%27%29%29" 
alt="Q(s,a) = V(s)+(A(s,a)-\frac{1}{|A|}\sum_{a'}A(s,a'))">  

