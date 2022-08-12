import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy as sp
from scipy.io import wavfile
class TrackedArray():
    def __init__(self , arr):
        self.arr = np.copy(arr)
        self.reset()
    def reset(self):
        self.indices = []
        self.values = []
        self.access_type = []
        self.full_copies = []
    def track(self , key , access_type):
        self.indices.append(key)
        self.values.append(self.arr[key])
        self.access_type.append(access_type)
        self.full_copies.append(np.copy(self.arr))
    def getActivity(self , idx = None):
        if isinstance(idx, type(None)):
            return [(i , op) for (i,op) in zip(self.indices,self.access_type)]
        else:
            return (self.indices[idx] , self.access_type[idx])
    def __getitem__(self , key):
        self.track(key , "get")
        return self.arr.__getitem__(key)
    def __setitem__(self , key , value):
        self.arr.__setitem__(key , value)
        self.track(key , "set")
    def __len__(self):
        return self.arr.__len__()
def freq_map(x,x_min=0,x_max=1000,freq_min=120,freq_max=1200):
    return np.interp(x,[x_min,x_max],[freq_min,freq_max])
def freq_sample(freq , dt=1./60. , samplerate = 44100 , oversample=2):
    mid_samples = np.int(dt * samplerate)
    pad_samples = np.int((mid_samples*(oversample-1)/2))
    total_samples = mid_samples + 2 * pad_samples
    y = np.sin(2*np.pi*freq*np.linspace(0,dt,np.int(dt * samplerate)))
    y[:pad_samples] = y[:pad_samples] * np.linspace(0, 1,pad_samples)
    y[- pad_samples:] = y[len(y) - pad_samples:]*\
        np.linspace(1,0,pad_samples)
    return y

plt.rcParams['figure.figsize'] = (12,8)
plt.rcParams['font.size'] =  16
FPS = 1000.0
# Audio Parameter
F_SAMPLE = 44100
OVERSAMPLE = 2
N = 20
arr = np.round(np.linspace(0,1000,N),0)
np.random.seed(0)
np.random.shuffle(arr)
arr = TrackedArray(arr)

# fig,ax = plt.subplots()
# ax.bar(np.arange(0,len(arr),1),arr,align='edge',width=0.8)
# ax.set_xlim([0,N])
# ax.set(xlabel='Index' , ylabel='Value' , title='Unsorted Array')
####### Insertion Sorts ###########
# sorter = 'Insertion'
# t = time.perf_counter()
# i = 1
# while (i < len(arr)):
#     j = i
#     while((j > 0) and arr[j-1] >  arr[j]):
#         temp = arr[j-1]
#         arr[j-1] = arr[j]
#         arr[j] = temp
#         j-=1
#     i+=1 
# dt = time.perf_counter() - t

################################################
################# Quick Sort ###################
sorter = 'Quick'
t = time.perf_counter()
def quickSort(current , low , high):
    if low < high:
        pivot = partition(current , low , high)
        quickSort(current, low, pivot-1)
        quickSort(current, pivot+1, high)
        
def partition(current , low , high):
    pivot = current[high]
    i = low
    for j in range(low , high):
        if current[j] < pivot:
            temp = current[i]
            current[i] = current[j]
            current[j] = temp
            i+=1
    temp = current[i]
    current[i] = current[high]
    current[high] = temp
    return i
t = time.perf_counter()
quickSort(arr, 0, len(arr) - 1)
dt = time.perf_counter() - t
# ##################################################
print(f"---------------{sorter} Sort -----------------")    
print(f"Array Sorted in {dt*1E3:.1f} ms")
# create sound file
wav_data = np.zeros(np.int(F_SAMPLE*len(arr.values)*1./FPS),dtype=
                    np.float)
dN = np.int(F_SAMPLE*1./FPS)
for i,value in enumerate(arr.values):	
    freq = freq_map(value)
    sample = freq_sample(freq,1./FPS,F_SAMPLE , oversample=OVERSAMPLE)
    idx_0 = np.int((i+0.5)*dN - len(sample)/2)
    idx_1 = idx_0 + len(sample)
    try:
        wav_data[idx_0:idx_1] += sample
    except ValueError:
        print(f"Failed to generate {i : .0f}th index sample")
    
sp.io.wavfile.write(f"{sorter}_sound.wav",F_SAMPLE , wav_data)
fig,ax = plt.subplots()
container = ax.bar(np.arange(0,len(arr),1),arr,align='edge',width=0.8)
ax.set_xlim([0,N])
txt = ax.text(0,1000,'')
def update(frame):
    txt.set_text(f"Accesses = {frame}")
    for(rectangle , height) in zip(container.patches,arr.full_copies[frame]):
        rectangle.set_height(height)
        rectangle.set_color('#1f77b4')
    idx,op = arr.getActivity(frame)
    if op == 'get':
        container.patches[idx].set_color('magenta')
    elif op == 'set':
        container.patches[idx].set_color('red')
    fig.savefig(f"frames/{sorter}_frame{frame:05.0f}.png")
    return (*container,txt)
ax.set(xlabel='Index' , ylabel='Value' , title=f'{sorter} Sort')
ani = FuncAnimation(fig, update, frames=range(len(arr.full_copies)),blit=True,
                    interval=100./FPS,repeat=False)