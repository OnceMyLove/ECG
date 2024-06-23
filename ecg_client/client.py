import matplotlib.pyplot as plt
from collections import deque
from socket import *
import struct
import time
from filter import *
HOST="192.168.4.1"
PORT=90
data_array=deque(maxlen=600)
t=0
client=socket(AF_INET,SOCK_STREAM)
client.connect((HOST,PORT))
plt.ion()
fig=plt.figure(figsize=(12,8))
ax = fig.add_subplot(211)
bx = fig.add_subplot(212)
count=50




while True:
    recv_data=client.recv(2*count)    #传到数据形如'490\r\n',完整放下需要4字节
    # print(recv_data)
    for i in range(count):
        unpack_data = struct.unpack('H', recv_data[i * 2:i * 2 + 2])[0]
        data_array.append(unpack_data)
    # print(data_array)
    
    
    ax.set_ylabel("ecg",fontdict=None, loc="top", x=100)
    ax.plot(data_array)
    filter_data=singnal_filter(data=data_array,frequency=count)
    filter_data=filter_data+data_array[0]
    ax.plot(filter_data)
    ax.legend(['original','filtered'],loc='upper right')
    bx.plot(data_array)
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.sca(ax)
    plt.cla()
    plt.sca(bx)
    plt.cla()
    time.sleep(0.5)
    

# if __name__=='__main__':
#     matplotlib_draw()
#     plt.show()
#     # client.close()