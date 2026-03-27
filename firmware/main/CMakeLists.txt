idf_component_register(SRCS "main.c" "fc.c"
                    INCLUDE_DIRS "."
                    PRIV_REQUIRES esp_timer esp_http_server esp32-camera esp_wifi nvs_flash esp_adc
                                  esp_driver_gpio esp_driver_ledc esp_driver_i2c)