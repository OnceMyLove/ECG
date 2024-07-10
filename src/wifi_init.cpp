
#include <wifi_init.h>


//ap模式设置
// const char* ssid = "ecg";
// const char* password = "123456789";
// const char host='192.168.4.1';

WiFiClient tcpclient;


//tcp模式设置
const char* ssid = "wyc";
const char* password = "qwertyuiop";
const char* serverIP = "172.20.10.11";//需要连接的目标，TCP服务器端地址。根据网段修改
WiFiServer server(90);
// const int serverPort = 90;
// WiFiClient esp_client;


// onenet云平台设置
const char* mqtt_server="mqtts.heclouds.com";
const int port=1883;
#define mqtt_deviceid "esp_ecg"
#define mqtt_productid "4YHmxpF4cV"
#define mqtt_password "version=2018-10-31&res=products%2F4YHmxpF4cV%2Fdevices%2Fesp_ecg&et=4118546126&method=md5&sign=Teu19xKIBCq9WJ8SZEGPZQ%3D%3D"
WiFiClient espclient;//创建客户端
PubSubClient mqttclient(espclient);//创建一个PubSub客户端，传入创建的WiFi客户端
char msgJson[75];//消息缓冲区
//信息模板
char dataTemplate[]="{\"id\":1,\"dp\":{\"result\":[{\"v\":%d}]}}";
Ticker tim1;

void callback(char *topic,byte *payload,uint8_t length);
void send();


uint8_t count=100;//采集的点数
uint16_t *array=(uint16_t*)malloc(count*sizeof(uint16_t));


void wifi_init(void){
    #if 1
    // WiFi.mode(WIFI_AP); // 设置为AP模式    
    // WiFi.softAP(ssid, password); // 创建WiFi接入点
    // IPAddress ip = WiFi.softAPIP(); // 获取AP的IP地址

    // for(int i=0;i<10;i++){    
    //         Serial.println();
    //         Serial.print("IP address: ");
    //         Serial.println(ip);

    //         delay(1000);
    //         }

    // STA相关
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid,password);
    
    while(WiFi.status()!=WL_CONNECTED)
    {
        Serial.println('.');
        delay(1000);
    }
    delay(5000);
    IPAddress ip = WiFi.localIP();
    Serial.println(ip);
    //启动TCP服务器
    server.begin();

    mqttclient.setServer(mqtt_server,port);//设置客户端，连接服务器
    mqttclient.connect(mqtt_deviceid,mqtt_productid,mqtt_password);
    //客户端连接到指定产品设备，同时输入鉴权信息
    while(!mqttclient.connected())
    {
        Serial.println("onenet is connecting!");
        delay(1000);
    }
    mqttclient.setCallback(callback);
    // mqttclient.subscribe("$sys/4YHmxpF4cV/esp32_wroom/cmd/request/#");
    tim1.attach(1,send);//单位:s
    Serial.println("init ok");
    #endif
    

}

void client(void){
    Serial.println(WiFi.localIP());
    WiFiClient Client=server.available();
    if(Client){
        Serial.println("new client");
        while(Client.connected()){
            for(int i=0;i<count;i++){   //一次采集count个点，然后上传
                array[i]=analogRead(A0);
                
                Serial.println(analogRead(A0));
                delay(10);   //ms
            }
            
            Client.write((uint8_t*)array,2*count);
            Serial.println(array[0],array[1]);
            
            
        }
        Client.stop();
        Serial.println("tcpclient disconnected");
    }


    // if(esp_client.connected())
    // {
    //     if(esp_client.available())
    //     {
    //         esp_client.println("send to sever");
    //     }
    // }
}



//回调函数 topic主题 payload 传过来的信息 length 长度
void callback(char *topic,byte *payload,uint8_t length){
 Serial.println("message rev:");
 Serial.println(topic);
 for(size_t i=0;i<length;i++)
 {
  Serial.print((char)payload[i]);
 }
 Serial.println();
}


//发送函数
void send()
{
  if(mqttclient.connected())
  {
    char* result="ok";
    int random_value=random(1,10);
    snprintf(msgJson,75,dataTemplate,random_value);
    
    Serial.print("public the data:");
    Serial.println(msgJson);
    // mqttclient.publish("$sys/4YHmxpF4cV/esp32_wroom/dp/post/json",
    // (uint8_t*)msgJson,strlen(msgJson));       //发送数据到主题

    mqttclient.publish("$sys/4YHmxpF4cV/esp_ecg/dp/post/json",
    msgJson); 
  }
}