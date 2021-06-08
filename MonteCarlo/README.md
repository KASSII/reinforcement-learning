# スクリプト概要
* モンテカルロ法でFrozenLakeを強化学習するスクリプト  

# 実行方法
## 学習
* 下記コマンドを実行  
`> python main.py`　　

* 学習が完了すると、学習したエージェントで1エピソード実行される  
* 実行するとlogフォルダ以下にタイムスタンプ名のフォルダが作成され、ログと学習結果が保存される

---
# モンテカルロ法の概要
## モンテカルロ法とは
* モデルフリーの価値ベース手法。  
* エージェントを動かし、実際に得られた報酬から状態価値を推定する。  
  エピソードを何度も繰り返すことで精度を上げる。  
  (要は繰り返し実行して期待値を求める方法)  

## モンテカルロ法の処理イメージ  
例）状態s_Aの状態価値をモンテカルロ法で推定する  
<b><u>1回目の試行</u></b>
<p align="left">
<img src="../docs/MonteCarlo/MonteCarlo_image1.jpg">
</p>

現在の推定状態価値  
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+V_%7Bestimate%7D%28s_A%29+%3D+0%0A" 
alt="V_{estimate}(s_A) = 0
">

実際に得られた状態価値  
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+V_%7Bgain%7D%28s_A%29+%3D+r_1+%2B+%5Cgamma+r_2" 
alt="V_{gain}(s_A) = r_1 + \gamma r_2">

更新した状態価値  
（今までの推定値と実際に得られた価値の平均）  
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+V_%7B1%7D%28s_A%29+%3D+%281-%5Calpha%29V_%7Bestimate%7D%28s_A%29+%2B+%5Calpha+V_%7Bgain%7D%28s_A%29" 
alt="V_{1}(s_A) = (1-\alpha)V_{estimate}(s_A) + \alpha V_{gain}(s_A)">


<b><u>2回目の試行</u></b>
<p align="left">
<img src="../docs/MonteCarlo/MonteCarlo_image2.jpg">
</p>

現在の推定状態価値  
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+V_%7Bestimate%7D%28s_A%29+%3D+V_%7B1%7D%28s_A%29" 
alt="V_{estimate}(s_A) = V_{1}(s_A)">

実際に得られた状態価値  
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+V_%7Bgain%7D%28s_A%29+%3D+r%27_1+%2B+%5Cgamma+r_3+%2B+%5Cgamma%5E2+r_4" 
alt="V_{gain}(s_A) = r'_1 + \gamma r_3 + \gamma^2 r_4">

更新した状態価値  
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+V_%7B2%7D%28s_A%29+%3D+%281-%5Calpha%29V_%7Bestimate%7D%28s_A%29+%2B+%5Calpha+V_%7Bgain%7D%28s_A%29" 
alt="V_{2}(s_A) = (1-\alpha)V_{estimate}(s_A) + \alpha V_{gain}(s_A)">


...　　


<b><u>N回目の試行</u></b>  
十分な回数を繰り返すことで確率が収束する  
<p align="left">
<img src="../docs/MonteCarlo/MonteCarlo_image3.jpg">
</p>

現在の推定状態価値  
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+V_%7Bestimate%7D%28s_A%29+%3D+V_%7BN-1%7D%28s_A%29" 
alt="V_{estimate}(s_A) = V_{N-1}(s_A)">

実際に得られた状態価値  
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+V_%7Bgain%7D%28s_A%29+%3D+r_1+%2B+%5Cgamma+r_5" 
alt="V_{gain}(s_A) = r_1 + \gamma r_5">

更新した状態価値  
<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+V_%7BN%7D%28s_A%29+%3D+%281-%5Calpha%29V_%7Bestimate%7D%28s_A%29+%2B+%5Calpha+V_%7Bgain%7D%28s_A%29" 
alt="V_{N}(s_A) = (1-\alpha)V_{estimate}(s_A) + \alpha V_{gain}(s_A)">


---
# 実装メモ  
* 観測した事がない状態の時に価値関数に従って行動選択すると選択が偏って学習できなくなるので、ランダムに行動選択させるようにする。  
（全ての行動価値の初期値は0にしているため、学習前の状態で値が大きい行動を取得しようとすると、リストの最初の行動しか選択されず学習ができなくなる）
* αは状態sの時に行動aが選ばれた回数（trainer.pyのN）を使用する。  