import numpy as np
from matplotlib import pyplot as plt

class Logger():
    def __init__(self):
        self.log = {}
    
    def add(self, data):
        for k, v in data.items():
            if k in self.log:
                self.log[k].append(v)
            else:
                self.log[k] = [v]
    
    def show_graph(self, key, freq):
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
        plt.plot(x, y)
        plt.show()
        return
        
