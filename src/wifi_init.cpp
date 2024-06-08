#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiAP.h>
#include <wifi_init.h>

const char* ssid = "ecg";
const char* password = "123456789";
const char host='192.168.4.1';
uint8_t count=100;//采集的点数
uint16_t *array=(uint16_t*)malloc(count*sizeof(uint16_t));

WiFiServer server(90);
void wifi_init(void){
    WiFi.mode(WIFI_AP); // 设置为AP模式
    WiFi.softAP(ssid, password); // 创建WiFi接入点
    IPAddress ip = WiFi.softAPIP(); // 获取AP的IP地址

    for(int i=0;i<10;i++){    
        Serial.println();
        Serial.print("AP IP address: ");
        Serial.println(ip);
        delay(1000);
        }

    
    //启动TCP服务器
    server.begin();
}

void client(void){
    WiFiClient Client=server.available();
    if(Client){
        Serial.println("new client");
        String data = Client.readStringUntil('\n');
        Serial.print("Received data: ");
        Serial.println(data);
        while(Client.connected()){
            for(int i=0;i<count;i++){
                array[i]=analogRead(A0);
                
                // Serial.println(analogRead(A0));
                delay(10);   
            }
            Client.write((uint8_t*)array,2*count);
            
        }
        Client.stop();
        Serial.println("client disconnected");
    }
}