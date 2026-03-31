const int ppgPin = A6;
const int respPin = A5;

void setup() {
  // Use a fast baud rate so data doesn't bottleneck!
  Serial.begin(115200); 
}

void loop() {
  unsigned long current_time = millis();
  
  // Read both analog pins
  int ppg_raw = analogRead(ppgPin);
  int resp_raw = analogRead(respPin);
  
  // Convert to true voltage (assuming 5V board, change to 3.3 if needed)
  float ppg_voltage = ppg_raw * (5.0 / 1023.0);
  float resp_voltage = resp_raw * (5.0 / 1023.0);

  // Print format: Time,PPG,Respiration
  Serial.print(current_time);
  Serial.print(",");
  Serial.print(ppg_voltage, 4);
  Serial.print(",");
  Serial.println(resp_voltage, 4);

  // 20ms delay = 50 Hz sampling rate. Perfect for both signals!
  delay(20); 
}
