import matplotlib.pyplot as plt
from collections import deque
from socket import *
import struct
import time
import datetime
import csv
from filter import *
from tensorflow.keras.models import load_model

HOST="172.20.10.2"
# HOST="0.0.0.0"
PORT=90
#采样数据规模相关
data_array=deque(maxlen=2000)
count=100   #一次获取200个点

client=socket(AF_INET,SOCK_STREAM)
client.connect((HOST,PORT))


plt.ion()
fig=plt.figure(figsize=(12,8))
ax = fig.add_subplot(211)
bx = fig.add_subplot(212)



def save_filename():#保存文件
    now_time=datetime.datetime.now()
    now_str=str(now_time)
    now_str=now_str.replace(":",".")
    now_str=now_str[:18]
    filename=r'./'+f'{now_str}.csv'
    return filename
    
        

filename=save_filename()
# #加载模型
model=load_model('final_model_1d_multihead_attention_gru_adamw_SMOTETEST.pb\saved_model.pb')
print(model.summary())

while True:
    
    
  
    recv_data=client.recv(2*count)    #传到数据形如'490\r\n',完整放下需要4字节
    print(recv_data[0:2])
    
    for i in range(count):
        unpack_data = struct.unpack('H', recv_data[i * 2:i * 2 + 2])[0]
        data_array.append(unpack_data)
            
    # f.flush()
    # print(data_array)
    ax.set_ylabel("ecg",fontdict=None, loc="top", x=100)
    ax.plot(data_array)
    filter_data=singnal_filter(data=data_array,frequency=count)
    filter_data=filter_data+data_array[0]
    # with open(filename,'a',newline='') as f:  #保存文件数据
    #     file_csv=csv.writer(f)
    #     file_csv.writerow([unpack_data])
    
    ax.plot(filter_data,'r')
    ax.legend(['original','filtered'],loc='upper right')
    bx.plot(filter_data,'r')
    bx.legend(['filtered'],loc='upper right')
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.sca(ax)
    plt.cla()
    plt.sca(bx)
    plt.cla()
    time.sleep(0.5)


# if __name__=='__main__':
#     import requests
#     import json
#     api_url=''
#     product_id='4YHmxpF4cV'
#     device_id='esp_ecg'
#     api_key='version=2018-10-31&res=products%2F4YHmxpF4cV%2Fdevices%2Fesp32_wroom&et=33262764145&method=md5&sign=oJs%2FCIVkTO4t%2F2Q5KEN2OQ%3D%3D'

#     data={
#         'datastreams':[
#             {
#                 'id':'ecg_result',
#                 'datapoints':[
#                     {
#                         'value':'健康'
#                     }
#                 ]
#             }
#         ]
#     }