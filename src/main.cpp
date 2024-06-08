#include <Arduino.h>
#include<wifi_init.h>
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
