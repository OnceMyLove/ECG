#include <Arduino.h>
#include <wifi_init.h>
//选择你烧录的开发板型号
#define xiao_esp32s3  //xiao_esp32s3    esp32dev

#if defined(xiao_esp32s3)
#define A0 6
#define A1 7
#define A2 8
void setup(){
  Serial.begin(9600);
  // pinMode(A0,INPUT);
  pinMode(A1,INPUT);
  pinMode(A2,INPUT);
  wifi_init();
  Serial.println("wifi setup");
}
void loop(){
  client();
  
  delay(1000);
}
#endif

#if defined(esp32dev)
#define A0 4
#define IO16 27
#define IO17 28
void setup(){
  Serial.begin(9600);
  // pinMode(A0,INPUT);
  pinMode(IO16,INPUT);
  pinMode(IO17,INPUT);
}
void loop(){
  if((digitalRead(IO16) == 1)||(digitalRead(IO17) == 1))
  {
  Serial.println('!');
  }
  else Serial.println(A0);
  delay(1000);
}
#endif




#if 0
const char* ssid = "wyc";
const char* password = "qwertyuiop";
const char* serverIP = "172.20.10.12";//需要连接的目标，TCP服务器端地址。根据网段修改
const int serverPort = 90;

uint8_t count=100;//采集的点数
uint16_t *array=(uint16_t*)malloc(count*sizeof(uint16_t));
WiFiClient espclient;

void setup() {
  Serial.begin(115200);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
  Serial.println("Connected");
    Serial.print("IP Address:");
    Serial.println(WiFi.localIP());//打印本机IP地址

  Serial.print("Connecting to server ");
  Serial.print(serverIP);
  Serial.print(":");
  Serial.println(serverPort);

  if (espclient.connect(serverIP, serverPort)) {
    Serial.println("Connected to server");
  } else {
    Serial.println("Connection failed");
  }
}

void loop() {
  if (espclient.connected()) {
    if (espclient.available()) 
    {
            for(int i=0;i<count;i++){   //一次采集count个点，然后上传
                
                array[i]=random(1,100);
                delay(10);   //ms
            
            
            espclient.write((uint8_t*)array,2*count);
            Serial.println(array[0],array[1]);
        
    }
    }
  } else {
    Serial.println("Connection lost");
    espclient.connect(serverIP, serverPort);
  }
}
#endif
    
    