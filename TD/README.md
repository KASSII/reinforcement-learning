# スクリプト概要　　
* TD法を実装したスクリプト  
* 実装済みのアルゴリズムは下記の通り  
  各アルゴリズムの詳細はそれぞれのフォルダを参照  
    * Sarsa  
    * Q-learning  
* 深層学習系列のアルゴリズムは別途実装

# TD法の概要  
## TD法とは  
* モンテカルロ法の欠点として、価値の更新に1エピソード完了する必要があるので学習に時間がかかる  
* TD法では見積もり（予測）の価値を使うことで1ステップごとに更新を行う  
<p align="left">
<img src="../docs/TD/TD.jpg">
</p>

実測と見積もりの誤差を**TD誤差**（Temporal Difference Error）という。  

<img src=
"https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+r%2B%5Cgamma+V%28s_%7Bt%2B1%7D%29-V%28s_t%29%0A" 
alt="r+\gamma V(s_{t+1})-V(s_t)
">

もし、時刻tにおける見積もり価値V(s_t)が真の価値V*(s_t)と同じならば、TD誤差は0になる。  
つまり、TD誤差が小さくなるように価値を修正していくのがTD法。  

<img src=
"https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+V%28s_t%29+%5CLeftarrow+V%28s_t%29+%2B+%5Calpha+%28r%2B%5Cgamma+V%28s_%7Bt%2B1%7D%29-V%28s_t%29%29" 
alt="V(s_t) \Leftarrow V(s_t) + \alpha (r+\gamma V(s_{t+1})-V(s_t))">

※実測を進めてV(s_t)を見積もり値ではなく実測値<img src=
"https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cgamma%28r_%7Bt%2B1%7D%2B%5Cgamma+r_%7Bt%2B2%7D+%2B+%5Cdots+%2B+%5Cgamma%5E%7Bn-1%7Dr_%7Bt%2Bn%7D%29" 
alt="\gamma(r_{t+1}+\gamma r_{t+2} + \dots + \gamma^{n-1}r_{t+n})">
を使うようにすると、モンテカルロ法と同義になる。