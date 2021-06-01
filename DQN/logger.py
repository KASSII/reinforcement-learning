import os
import datetime
import json
import numpy as np
from matplotlib import pyplot as plt

class Logger():
    def __init__(self, alg_name, output_log_file="Log.txt"):
        self.alg_name = alg_name
        self.output_log_file = output_log_file
        self.log = {}
    
    # 初期化
    def initialize(self, title="", config={}):
        os.makedirs(os.path.dirname(self.output_log_file), exist_ok=True)
        with open(self.output_log_file, "w") as f:
            f.write("{}\n".format(title))
            f.write("Algorithm: {}\n".format(self.alg_name))
            f.write("config: {}\n".format(config))
            f.write("\n")
        self.log = {}

    # ログデータを追加
    def add(self, data):
        for k, v in data.items():
            if k in self.log:
                self.log[k].append(v)
            else:
                self.log[k] = [v]
        
        # ログファイルに出力
        with open(self.output_log_file, "a") as f:
            now = datetime.datetime.now()
            f.write("{}\t\t{}\n".format(now.strftime('%Y-%m-%d %H:%M:%S'), str(data)))
    
    # 直近latest分のkeyの値を取得
    def get(self, key, latest=1):
        if not key in self.log:
            return None
        
        target_log = self.log[key]
        ave = np.array(target_log[-1*latest:]).mean()
        return ave

    # グラフを表示
    def plot(self, key, freq, save_path=None):
        if not key in self.log:
            print("{} is not registered.".format(key))
            return
        
        target_log = self.log[key]
        x = []
        y = []
        for i in range(int(len(target_log)/freq)):
            ave = np.array(target_log[i*freq:(i+1)*freq]).mean()
            x.append(i*freq)
            y.append(ave)
        plt.title(key)
        plt.plot(x, y)

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
        return
        
