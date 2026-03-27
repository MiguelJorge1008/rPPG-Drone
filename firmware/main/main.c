#include <esp_system.h>
#include <nvs_flash.h>
#include <sys/param.h>
#include <string.h>
#include <stdlib.h>
#include "esp_timer.h"
#include "esp_camera.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_http_server.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "driver/gpio.h"
#include "esp_sleep.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "fc.h"

static const char *TAG = "XIAO_S3_DRONE";

/* ==========================================================
   BATTERY
   ========================================================== */
#define BAT_ADC_CHANNEL   ADC_CHANNEL_0   // GPIO 1 (D0)
#define BAT_WARN_MV       3500
#define BAT_CRITICAL_MV   3000
#define LED_GPIO          GPIO_NUM_21

static adc_oneshot_unit_handle_t adc1_handle;
static adc_cali_handle_t adc_cali_handle = NULL;

static void bat_adc_init(void) {
    adc_oneshot_unit_init_cfg_t init_cfg = { .unit_id = ADC_UNIT_1 };
    adc_oneshot_new_unit(&init_cfg, &adc1_handle);

    adc_oneshot_chan_cfg_t chan_cfg = {
        .atten    = ADC_ATTEN_DB_12,
        .bitwidth = ADC_BITWIDTH_12,
    };
    adc_oneshot_config_channel(adc1_handle, BAT_ADC_CHANNEL, &chan_cfg);

    adc_cali_curve_fitting_config_t cali_cfg = {
        .unit_id  = ADC_UNIT_1,
        .chan     = BAT_ADC_CHANNEL,
        .atten    = ADC_ATTEN_DB_12,
        .bitwidth = ADC_BITWIDTH_12,
    };
    if (adc_cali_create_scheme_curve_fitting(&cali_cfg, &adc_cali_handle) != ESP_OK) {
        ESP_LOGW(TAG, "ADC calibration unavailable");
        adc_cali_handle = NULL;
    }

    gpio_config_t led_cfg = {
        .pin_bit_mask = (1ULL << LED_GPIO),
        .mode         = GPIO_MODE_OUTPUT,
    };
    gpio_config(&led_cfg);
    gpio_set_level(LED_GPIO, 0);
}

static int bat_read_mv(void) {
    int sum = 0;
    for (int i = 0; i < 8; i++) {
        int raw = 0;
        adc_oneshot_read(adc1_handle, BAT_ADC_CHANNEL, &raw);
        int voltage_mv = 0;
        if (adc_cali_handle) {
            adc_cali_raw_to_voltage(adc_cali_handle, raw, &voltage_mv);
        } else {
            voltage_mv = (int)((raw / 4095.0f) * 3100.0f);
        }
        sum += voltage_mv;
    }
    return (sum / 8) * 2; // average of 8 readings, 100K+100K voltage divider
}

static void battery_task(void *arg) {
    int critical_count = 0;
    while (1) {
        int mv = bat_read_mv();
        ESP_LOGI(TAG, "Battery: %d mV", mv);

        if (mv < 2000) {
            ESP_LOGI(TAG, "No battery detected (%d mV)", mv);
            critical_count = 0;
        } else if (mv < BAT_CRITICAL_MV) {
            critical_count++;
            ESP_LOGW(TAG, "Low battery (%d mV) %d/3", mv, critical_count);
            if (critical_count >= 3) {
                ESP_LOGW(TAG, "Critical battery! Entering deep sleep...");
                fc_set_state(FC_DISARMED);
                for (int i = 0; i < 10; i++) {
                    gpio_set_level(LED_GPIO, i % 2);
                    vTaskDelay(pdMS_TO_TICKS(100));
                }
                gpio_set_level(LED_GPIO, 0);
                esp_deep_sleep_start();
            }
        } else if (mv < BAT_WARN_MV) {
            critical_count = 0;
            for (int i = 0; i < 6; i++) {
                gpio_set_level(LED_GPIO, i % 2);
                vTaskDelay(pdMS_TO_TICKS(150));
            }
        } else {
            critical_count = 0;
            gpio_set_level(LED_GPIO, 0);
        }

        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}

/* ==========================================================
   WI-FI CONFIGURATION
   ========================================================== */
#define WIFI_SSID      "rPPG-Drone"
#define WIFI_PASS      "12345678"

/* ==========================================================
   XIAO ESP32-S3 CAMERA GPIO MAPPING
   ========================================================== */
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  10
#define SIOD_GPIO_NUM  40
#define SIOC_GPIO_NUM  39

#define Y9_GPIO_NUM    48
#define Y8_GPIO_NUM    11
#define Y7_GPIO_NUM    12
#define Y6_GPIO_NUM    14
#define Y5_GPIO_NUM    16
#define Y4_GPIO_NUM    18
#define Y3_GPIO_NUM    17
#define Y2_GPIO_NUM    15

#define VSYNC_GPIO_NUM 38
#define HREF_GPIO_NUM  47
#define PCLK_GPIO_NUM  13

/* MJPEG boundary */
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

/* ==========================================================
   HTML PAGE
   ========================================================== */
static const char INDEX_HTML[] =
"<!DOCTYPE html><html><head>"
"<meta charset='utf-8'>"
"<meta name='viewport' content='width=device-width,initial-scale=1'>"
"<title>rPPG Drone</title>"
"</head><body style='background:#111;color:#eee;font-family:monospace;text-align:center;padding:10px'>"
"<h3 style='margin:4px 0'>rPPG Drone Control</h3>"

/* ── Camera controls ── */
"<p style='margin:6px 0 2px'>AEC (0-1200): <b id='va'>1200</b></p>"
"<input type='range' min='0' max='1200' value='1200' style='width:80%' "
"oninput=\"document.getElementById('va').innerText=this.value;aecVal=this.value\">"
"<p style='margin:6px 0 2px'>Gain (0-30): <b id='vg'>3</b></p>"
"<input type='range' min='0' max='30' value='3' style='width:80%' "
"oninput=\"document.getElementById('vg').innerText=this.value;gainVal=this.value\">"
"<button onclick='sendCam()' style='display:block;margin:8px auto;padding:8px 28px;background:#1a7a1a;color:#fff;border:none;font-size:14px;cursor:pointer'>Apply Camera</button>"
"<p id='st' style='color:#aaa;font-size:11px;margin:2px 0 6px'></p>"

/* ── Stream ── */
"<img id='si' src='http://192.168.4.1:81/stream' style='max-width:100%;display:block;margin:0 auto;border:1px solid #444'>"

/* ── IMU / Attitude ── */
"<div style='margin:10px auto;max-width:400px;background:#1e1e1e;border:1px solid #333;padding:8px;text-align:left;font-size:12px'>"
"<b>IMU</b> &nbsp;<span id='imu_err' style='color:#f66'></span><br>"
"Accel: X=<span id='ax'>-</span>g Y=<span id='ay'>-</span>g Z=<span id='az'>-</span>g<br>"
"Gyro: X=<span id='gx'>-</span> Y=<span id='gy'>-</span> Z=<span id='gz'>-</span> deg/s<br>"
"<b>Attitude:</b> Roll=<span id='att_r'>-</span>&deg; Pitch=<span id='att_p'>-</span>&deg; Yaw=<span id='att_y'>-</span>&deg;<br>"
"<b>Battery:</b> <span id='bat'>-</span> mV"
"</div>"

/* ── Flight controls ── */
"<div style='margin:10px auto;max-width:400px;background:#1e1e1e;border:1px solid #333;padding:8px;font-size:12px'>"
"<b>Flight Control</b><br>"
"State: <b id='fc_state' style='color:#ff0'>DISARMED</b><br><br>"
"<button onclick='fcCmd(\"state=arm\")' style='padding:6px 16px;background:#7a1a1a;color:#fff;border:none;cursor:pointer;margin:3px'>ARM</button>"
"<button onclick='fcCmd(\"state=disarm\")' style='padding:6px 16px;background:#444;color:#fff;border:none;cursor:pointer;margin:3px'>DISARM</button>"
"<p style='color:#666;font-size:10px;margin:4px 0'>ARM enables manual /cmd?roll=&amp;pitch=&amp;yaw=&amp;thrust=</p>"
"</div>"

/* ── Joystick ── */
"<div style='margin:10px auto;max-width:400px'>"
"<div style='display:flex;justify-content:space-around;align-items:center'>"
"<div style='text-align:center'>"
"<p style='margin:2px;font-size:10px;color:#aaa'>Thrust / Yaw</p>"
"<canvas id='lj' width='130' height='130' style='border:2px solid #555;border-radius:50%;background:#1a1a1a;touch-action:none'></canvas>"
"</div>"
"<div style='text-align:center'>"
"<p style='margin:2px;font-size:10px;color:#aaa'>Pitch / Roll</p>"
"<canvas id='rj' width='130' height='130' style='border:2px solid #555;border-radius:50%;background:#1a1a1a;touch-action:none'></canvas>"
"</div>"
"</div>"
"<p id='ws_st' style='font-size:10px;color:#f66;margin:4px 0;text-align:center'>WS desligado</p>"
"</div>"

/* ── PID tuning ── */
"<div style='margin:10px auto;max-width:400px;background:#1e1e1e;border:1px solid #333;padding:8px;font-size:11px;text-align:left'>"
"<b>PID Tuning</b> <button onclick='loadPid()' style='float:right;padding:2px 8px;font-size:10px;background:#333;color:#eee;border:1px solid #555;cursor:pointer'>Load</button><br><br>"
"Att kp:<input id='att_kp' style='width:55px;background:#111;color:#eee;border:1px solid #444;margin:2px' value='3.5'> "
"ki:<input id='att_ki' style='width:55px;background:#111;color:#eee;border:1px solid #444;margin:2px' value='2.0'><br>"
"Rate R kp:<input id='rr_kp' style='width:50px;background:#111;color:#eee;border:1px solid #444;margin:2px' value='70'> "
"ki:<input id='rr_ki' style='width:45px;background:#111;color:#eee;border:1px solid #444;margin:2px' value='0'> "
"kd:<input id='rr_kd' style='width:45px;background:#111;color:#eee;border:1px solid #444;margin:2px' value='0'><br>"
"Rate P kp:<input id='rp_kp' style='width:50px;background:#111;color:#eee;border:1px solid #444;margin:2px' value='70'> "
"ki:<input id='rp_ki' style='width:45px;background:#111;color:#eee;border:1px solid #444;margin:2px' value='0'> "
"kd:<input id='rp_kd' style='width:45px;background:#111;color:#eee;border:1px solid #444;margin:2px' value='0'><br>"
"Rate Y kp:<input id='ry_kp' style='width:50px;background:#111;color:#eee;border:1px solid #444;margin:2px' value='50'> "
"ki:<input id='ry_ki' style='width:45px;background:#111;color:#eee;border:1px solid #444;margin:2px' value='0'> "
"kd:<input id='ry_kd' style='width:45px;background:#111;color:#eee;border:1px solid #444;margin:2px' value='0'><br>"
"<button onclick='applyPid()' style='margin-top:6px;padding:4px 16px;background:#1a4a7a;color:#fff;border:none;cursor:pointer'>Apply PID</button>"
"<span id='pid_st' style='color:#aaa;font-size:10px;margin-left:8px'></span>"
"</div>"

/* ── Scripts ── */
"<script>"
"var aecVal=1200,gainVal=3;"
"function sendCam(){"
"var img=document.getElementById('si');"
"img.src='';"
"var x=new XMLHttpRequest();"
"x.open('GET','/control?aec='+aecVal+'&gain='+gainVal);"
"x.onload=function(){"
"document.getElementById('st').innerText=x.responseText;"
"setTimeout(function(){img.src='http://192.168.4.1:81/stream?t='+Date.now();},200);};"
"x.onerror=function(){document.getElementById('st').innerText='Error';"
"img.src='http://192.168.4.1:81/stream';};"
"x.send();}"

"function fcCmd(q){"
"var x=new XMLHttpRequest();"
"x.open('GET','/cmd?'+q);"
"x.onload=function(){"
"try{"
"var d=JSON.parse(x.responseText);"
"document.getElementById('fc_state').innerText=d.state.toUpperCase();"
"}catch(e){}"
"};"
"x.send();}"


"function updateImu(){"
"var x=new XMLHttpRequest();"
"x.open('GET','/imu');"
"x.onload=function(){"
"try{"
"var d=JSON.parse(x.responseText);"
"document.getElementById('ax').innerText=d.ax.toFixed(2);"
"document.getElementById('ay').innerText=d.ay.toFixed(2);"
"document.getElementById('az').innerText=d.az.toFixed(2);"
"document.getElementById('gx').innerText=d.gx.toFixed(1);"
"document.getElementById('gy').innerText=d.gy.toFixed(1);"
"document.getElementById('gz').innerText=d.gz.toFixed(1);"
"document.getElementById('att_r').innerText=d.roll.toFixed(1);"
"document.getElementById('att_p').innerText=d.pitch.toFixed(1);"
"document.getElementById('att_y').innerText=d.yaw.toFixed(1);"
"document.getElementById('bat').innerText=d.bat;"
"var states={'disarmed':'DISARMED','flight':'FLIGHT','hover':'HOVER+rPPG'};"
"if(d.fc_state) document.getElementById('fc_state').innerText=states[d.fc_state]||d.fc_state.toUpperCase();"
"document.getElementById('imu_err').innerText='';"
"}catch(e){document.getElementById('imu_err').innerText='error';}};"
"x.onerror=function(){document.getElementById('imu_err').innerText='offline';};"
"x.send();}"
"setInterval(updateImu,500);"
"updateImu();"

/* WebSocket + Joystick JS */
"var ws=null,wsOk=false;"
"var lx=0,ly=0.5,rx=0,ry=0;"  /* ly=0.5 → thrust começa a 50% */

"function wsConn(){"
"ws=new WebSocket('ws://192.168.4.1/ws');"
"ws.onopen=function(){wsOk=true;document.getElementById('ws_st').style.color='#4a4';document.getElementById('ws_st').innerText='WS ligado';};"
"ws.onclose=function(){wsOk=false;document.getElementById('ws_st').style.color='#f66';document.getElementById('ws_st').innerText='WS desligado';setTimeout(wsConn,2000);};"
"ws.onerror=function(){ws.close();};"
"}"
"wsConn();"

/* Envia setpoint a 20 Hz via WebSocket */
"setInterval(function(){"
"if(wsOk&&ws.readyState===1){"
"var r=(rx*30).toFixed(1);"
"var p=(-ry*30).toFixed(1);"
"var y=(lx*200).toFixed(1);"
"var t=((ly+1)/2).toFixed(2);"  /* [-1,1] → [0,1] */
"ws.send('{\"r\":'+r+',\"p\":'+p+',\"y\":'+y+',\"t\":'+t+'}');"
"}},50);"

/* Joystick helper */
"function mkJoy(id,isLeft){"
"var c=document.getElementById(id),ctx=c.getContext('2d');"
"var R=c.width/2,r=R-8,tx=R,ty=R,act=false;"
"function draw(){"
"ctx.clearRect(0,0,c.width,c.height);"
"ctx.beginPath();ctx.arc(R,R,r,0,6.28);ctx.strokeStyle='#555';ctx.lineWidth=1;ctx.stroke();"
"ctx.beginPath();ctx.arc(R,R,2,0,6.28);ctx.fillStyle='#555';ctx.fill();"
"ctx.beginPath();ctx.arc(tx,ty,18,0,6.28);ctx.fillStyle=wsOk?'#4a4':'#555';ctx.fill();}"
"function pos(ex,ey){"
"var rc=c.getBoundingClientRect(),px=ex-rc.left-R,py=ey-rc.top-R;"
"var d=Math.sqrt(px*px+py*py);if(d>r){px=px/d*r;py=py/d*r;}"
"tx=R+px;ty=R+py;"
"if(isLeft){lx=px/r;ly=-(py/r);}else{rx=px/r;ry=py/r;}"
"draw();}"
"c.addEventListener('touchstart',function(e){e.preventDefault();act=true;pos(e.touches[0].clientX,e.touches[0].clientY);},{passive:false});"
"c.addEventListener('touchmove',function(e){e.preventDefault();if(act)pos(e.touches[0].clientX,e.touches[0].clientY);},{passive:false});"
"c.addEventListener('touchend',function(e){e.preventDefault();act=false;"
"if(isLeft){lx=0;tx=R;ty=R-ly*r;}"  /* yaw centra, thrust mantém */
"else{rx=0;ry=0;tx=R;ty=R;}"         /* pitch/roll centram */
"draw();},{passive:false});"
"setInterval(draw,200);"  /* redraw periódico para atualizar cor WS */
"draw();}"
"mkJoy('lj',true);mkJoy('rj',false);"

/* PID tuning */
"function loadPid(){"
"fetch('/pid').then(function(r){return r.json();}).then(function(d){"
"document.getElementById('att_kp').value=d.att_kp;"
"document.getElementById('att_ki').value=d.att_ki;"
"document.getElementById('rr_kp').value=d.rr_kp;"
"document.getElementById('rr_ki').value=d.rr_ki;"
"document.getElementById('rr_kd').value=d.rr_kd;"
"document.getElementById('rp_kp').value=d.rp_kp;"
"document.getElementById('rp_ki').value=d.rp_ki;"
"document.getElementById('rp_kd').value=d.rp_kd;"
"document.getElementById('ry_kp').value=d.ry_kp;"
"document.getElementById('ry_ki').value=d.ry_ki;"
"document.getElementById('ry_kd').value=d.ry_kd;"
"document.getElementById('pid_st').innerText='OK';});}"

"function applyPid(){"
"var q='att_kp='+document.getElementById('att_kp').value"
"+'&att_ki='+document.getElementById('att_ki').value"
"+'&rr_kp='+document.getElementById('rr_kp').value"
"+'&rr_ki='+document.getElementById('rr_ki').value"
"+'&rr_kd='+document.getElementById('rr_kd').value"
"+'&rp_kp='+document.getElementById('rp_kp').value"
"+'&rp_ki='+document.getElementById('rp_ki').value"
"+'&rp_kd='+document.getElementById('rp_kd').value"
"+'&ry_kp='+document.getElementById('ry_kp').value"
"+'&ry_ki='+document.getElementById('ry_ki').value"
"+'&ry_kd='+document.getElementById('ry_kd').value;"
"fetch('/pid?'+q).then(function(){document.getElementById('pid_st').innerText='Applied';});}"

"</script>"
"</body></html>";

/* ==========================================================
   HTTP HANDLERS
   ========================================================== */

/* GET /imu — IMU raw + attitude + battery JSON */
esp_err_t imu_handler(httpd_req_t *req) {
    char buf[220];
    int bat = bat_read_mv();

    fc_imu_t imu;
    fc_get_imu(&imu);

    float roll, pitch, yaw;
    fc_get_attitude(&roll, &pitch, &yaw);

    const char *state_str =
        fc_get_state() == FC_DISARMED      ? "disarmed" :
        fc_get_state() == FC_FLIGHT        ? "flight"   : "hover";

    snprintf(buf, sizeof(buf),
        "{\"ax\":%.3f,\"ay\":%.3f,\"az\":%.3f,"
        "\"gx\":%.2f,\"gy\":%.2f,\"gz\":%.2f,"
        "\"roll\":%.1f,\"pitch\":%.1f,\"yaw\":%.1f,"
        "\"bat\":%d,\"fc_state\":\"%s\"}",
        imu.ax, imu.ay, imu.az,
        imu.gx, imu.gy, imu.gz,
        roll, pitch, yaw,
        bat, state_str);

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, buf, strlen(buf));
}

/* GET / — main HTML page */
esp_err_t index_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
}

/* GET /stream — MJPEG stream */
esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t * fb = NULL;
    esp_err_t res = ESP_OK;
    char part_buf[64];

    static int64_t last_frame_time = 0;
    static uint32_t frames = 0;

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if(res != ESP_OK) return res;

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    while(true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "Camera capture failed");
            res = ESP_FAIL;
        } else {
            frames++;
            int64_t now = esp_timer_get_time();
            if (now - last_frame_time >= 1000000) {
                float fps = (frames * 1000000.0) / (now - last_frame_time);
                printf(">>> STREAMING: %.2f FPS | Res: QVGA\n", fps);
                frames = 0;
                last_frame_time = now;
            }

            size_t hlen = snprintf(part_buf, 64, _STREAM_PART, fb->len);
            res = httpd_resp_send_chunk(req, part_buf, hlen);
            if(res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
            if(res == ESP_OK) res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
            esp_camera_fb_return(fb);
        }
        if(res != ESP_OK) break;
    }
    return res;
}

/* GET /control?aec=X&gain=Y — camera exposure/gain */
esp_err_t control_handler(httpd_req_t *req) {
    char query[64];
    char param[16];
    char resp[64];

    sensor_t *s = esp_camera_sensor_get();
    if (!s) {
        httpd_resp_send(req, "Error: sensor not available", -1);
        return ESP_FAIL;
    }

    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK) {
        httpd_resp_send(req, "No parameters", -1);
        return ESP_OK;
    }

    int aec = -1, gain = -1;

    if (httpd_query_key_value(query, "aec", param, sizeof(param)) == ESP_OK) {
        aec = atoi(param);
        if (aec < 0)    aec = 0;
        if (aec > 1200) aec = 1200;
        s->set_aec_value(s, aec);
    }

    if (httpd_query_key_value(query, "gain", param, sizeof(param)) == ESP_OK) {
        gain = atoi(param);
        if (gain < 0)  gain = 0;
        if (gain > 30) gain = 30;
        s->set_agc_gain(s, gain);
    }

    snprintf(resp, sizeof(resp), "OK | aec=%d gain=%d", aec, gain);
    ESP_LOGI(TAG, "Control: %s", resp);

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, resp, strlen(resp));
}

/* GET /cmd?state=arm|disarm|hover[&thrust=0.4]
 *       OR
 * GET /cmd?roll=0&pitch=0&yaw=0&thrust=0.5
 */
esp_err_t cmd_handler(httpd_req_t *req) {
    char query[128];
    char param[32];
    char resp[160];

    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK) {
        httpd_resp_set_type(req, "application/json");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        httpd_resp_send(req, "{\"ok\":false,\"err\":\"no params\"}", -1);
        return ESP_OK;
    }

    if (httpd_query_key_value(query, "state", param, sizeof(param)) == ESP_OK) {
        /* ── State transition ── */
        if (strcmp(param, "arm") == 0) {
            fc_set_state(FC_FLIGHT);
        } else if (strcmp(param, "disarm") == 0) {
            fc_set_state(FC_DISARMED);
        }
    } else {
        /* ── Setpoint ── */
        fc_setpoint_t sp = {0};

        if (httpd_query_key_value(query, "roll",   param, sizeof(param)) == ESP_OK)
            sp.roll   = strtof(param, NULL);
        if (httpd_query_key_value(query, "pitch",  param, sizeof(param)) == ESP_OK)
            sp.pitch  = strtof(param, NULL);
        if (httpd_query_key_value(query, "yaw",    param, sizeof(param)) == ESP_OK)
            sp.yaw    = strtof(param, NULL);
        if (httpd_query_key_value(query, "thrust", param, sizeof(param)) == ESP_OK)
            sp.thrust = strtof(param, NULL);

        /* Clamp */
        if (sp.roll   >  30.0f) sp.roll   =  30.0f;
        if (sp.roll   < -30.0f) sp.roll   = -30.0f;
        if (sp.pitch  >  30.0f) sp.pitch  =  30.0f;
        if (sp.pitch  < -30.0f) sp.pitch  = -30.0f;
        if (sp.yaw    > 200.0f) sp.yaw    = 200.0f;
        if (sp.yaw    <-200.0f) sp.yaw    =-200.0f;
        if (sp.thrust >   1.0f) sp.thrust =   1.0f;
        if (sp.thrust <   0.0f) sp.thrust =   0.0f;

        fc_set_setpoint(&sp);
    }

    /* Response */
    float roll, pitch, yaw;
    fc_get_attitude(&roll, &pitch, &yaw);
    const char *state_str =
        fc_get_state() == FC_DISARMED      ? "disarmed" :
        fc_get_state() == FC_FLIGHT        ? "flight"   : "hover";

    snprintf(resp, sizeof(resp),
        "{\"ok\":true,\"state\":\"%s\","
        "\"roll\":%.1f,\"pitch\":%.1f,\"yaw\":%.1f}",
        state_str, roll, pitch, yaw);

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, resp, strlen(resp));
}

/* ==========================================================
   WEBSOCKET HANDLER  (/ws)
   ========================================================== */

/* Extract float value for a given key from a compact JSON string */
static float json_float(const char *json, const char *key, float def)
{
    char needle[16];
    snprintf(needle, sizeof(needle), "\"%s\":", key);
    const char *p = strstr(json, needle);
    if (!p) return def;
    return strtof(p + strlen(needle), NULL);
}

esp_err_t ws_handler(httpd_req_t *req)
{
    if (req->method == HTTP_GET) {
        /* WebSocket handshake — nothing to do */
        return ESP_OK;
    }

    httpd_ws_frame_t pkt;
    memset(&pkt, 0, sizeof(pkt));
    pkt.type = HTTPD_WS_TYPE_TEXT;

    /* First call: get frame length */
    esp_err_t ret = httpd_ws_recv_frame(req, &pkt, 0);
    if (ret != ESP_OK || pkt.len == 0) return ret;

    uint8_t *buf = calloc(1, pkt.len + 1);
    if (!buf) return ESP_ERR_NO_MEM;
    pkt.payload = buf;

    ret = httpd_ws_recv_frame(req, &pkt, pkt.len);
    if (ret == ESP_OK && pkt.type == HTTPD_WS_TYPE_TEXT) {
        fc_setpoint_t sp;
        sp.roll   = json_float((char *)buf, "r", 0.0f);
        sp.pitch  = json_float((char *)buf, "p", 0.0f);
        sp.yaw    = json_float((char *)buf, "y", 0.0f);
        sp.thrust = json_float((char *)buf, "t", 0.0f);

        if (sp.roll   >  30.0f) sp.roll   =  30.0f;
        if (sp.roll   < -30.0f) sp.roll   = -30.0f;
        if (sp.pitch  >  30.0f) sp.pitch  =  30.0f;
        if (sp.pitch  < -30.0f) sp.pitch  = -30.0f;
        if (sp.yaw    > 200.0f) sp.yaw    = 200.0f;
        if (sp.yaw    <-200.0f) sp.yaw    =-200.0f;
        if (sp.thrust >   1.0f) sp.thrust =   1.0f;
        if (sp.thrust <   0.0f) sp.thrust =   0.0f;

        fc_set_setpoint(&sp);
    }

    free(buf);
    return ret;
}

/* ==========================================================
   PID HANDLER  (GET /pid?att_kp=3.5&rr_kp=70&...)
   ========================================================== */
esp_err_t pid_handler(httpd_req_t *req)
{
    char query[256];
    char param[32];
    fc_pid_cfg_t cfg;
    fc_get_pid(&cfg);

    if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK) {
#define GETF(key, field) \
        if (httpd_query_key_value(query, key, param, sizeof(param)) == ESP_OK) \
            cfg.field = strtof(param, NULL);
        GETF("att_kp",  att_kp)
        GETF("att_ki",  att_ki)
        GETF("rr_kp",   rate_roll_kp)
        GETF("rr_ki",   rate_roll_ki)
        GETF("rr_kd",   rate_roll_kd)
        GETF("rp_kp",   rate_pitch_kp)
        GETF("rp_ki",   rate_pitch_ki)
        GETF("rp_kd",   rate_pitch_kd)
        GETF("ry_kp",   rate_yaw_kp)
        GETF("ry_ki",   rate_yaw_ki)
        GETF("ry_kd",   rate_yaw_kd)
#undef GETF
        fc_set_pid(&cfg);
        fc_get_pid(&cfg);
    }

    char resp[320];
    snprintf(resp, sizeof(resp),
        "{\"att_kp\":%.2f,\"att_ki\":%.3f,"
        "\"rr_kp\":%.1f,\"rr_ki\":%.3f,\"rr_kd\":%.3f,"
        "\"rp_kp\":%.1f,\"rp_ki\":%.3f,\"rp_kd\":%.3f,"
        "\"ry_kp\":%.1f,\"ry_ki\":%.3f,\"ry_kd\":%.3f}",
        cfg.att_kp, cfg.att_ki,
        cfg.rate_roll_kp, cfg.rate_roll_ki, cfg.rate_roll_kd,
        cfg.rate_pitch_kp, cfg.rate_pitch_ki, cfg.rate_pitch_kd,
        cfg.rate_yaw_kp,  cfg.rate_yaw_ki,  cfg.rate_yaw_kd);

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, resp, strlen(resp));
}

/* ==========================================================
   CAMERA INIT
   ========================================================== */
esp_err_t init_camera() {
    camera_config_t config = {
        .pin_pwdn = PWDN_GPIO_NUM,
        .pin_reset = RESET_GPIO_NUM,
        .pin_xclk = XCLK_GPIO_NUM,
        .pin_sccb_sda = SIOD_GPIO_NUM,
        .pin_sccb_scl = SIOC_GPIO_NUM,
        .pin_d7 = Y9_GPIO_NUM, .pin_d6 = Y8_GPIO_NUM, .pin_d5 = Y7_GPIO_NUM,
        .pin_d4 = Y6_GPIO_NUM, .pin_d3 = Y5_GPIO_NUM, .pin_d2 = Y4_GPIO_NUM,
        .pin_d1 = Y3_GPIO_NUM, .pin_d0 = Y2_GPIO_NUM,
        .pin_vsync = VSYNC_GPIO_NUM, .pin_href = HREF_GPIO_NUM, .pin_pclk = PCLK_GPIO_NUM,

        .xclk_freq_hz = 24000000,
        .ledc_timer   = LEDC_TIMER_0,    /* camera keeps TIMER_0 */
        .ledc_channel = LEDC_CHANNEL_0,  /* camera keeps CHANNEL_0 */
        .pixel_format = PIXFORMAT_JPEG,
        .frame_size   = FRAMESIZE_QVGA,
        .jpeg_quality = 8,
        .fb_count     = 3,
        .grab_mode    = CAMERA_GRAB_LATEST,
        .fb_location  = CAMERA_FB_IN_PSRAM
    };

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) return err;

    sensor_t *s = esp_camera_sensor_get();

    s->set_vflip(s, 1);
    s->set_hmirror(s, 1);
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_quality(s, 8);

    /* KILL AUTOMATICS (critical for rPPG) */
    s->set_whitebal(s, 0);
    s->set_gain_ctrl(s, 0);
    s->set_exposure_ctrl(s, 0);
    s->set_aec2(s, 0);

    s->set_agc_gain(s, 3);
    s->set_aec_value(s, 1200);

    return ESP_OK;
}

/* ==========================================================
   WI-FI EVENT HANDLER
   ========================================================== */
static void event_handler(void* arg, esp_event_base_t event_base,
                           int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_AP_STACONNECTED) {
        ESP_LOGI(TAG, "Client connected to AP");
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_AP_STADISCONNECTED) {
        ESP_LOGI(TAG, "Client disconnected from AP");
    }
}

/* ==========================================================
   APP MAIN
   ========================================================== */
void app_main(void) {
    esp_log_level_set("cam_hal", ESP_LOG_NONE);
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_LOGI(TAG, "--- Step 0: Battery Monitor ---");
    bat_adc_init();
    xTaskCreate(battery_task, "battery", 3072, NULL, 3, NULL);

    ESP_LOGI(TAG, "--- Step 1: Flight Controller (IMU + Motors) ---");
    fc_init();

    ESP_LOGI(TAG, "--- Step 2: Camera ---");
    if(init_camera() != ESP_OK) {
        ESP_LOGE(TAG, "Fatal camera error. Restarting in 5s...");
        vTaskDelay(pdMS_TO_TICKS(5000));
        esp_restart();
    }

    vTaskDelay(pdMS_TO_TICKS(2000));

    ESP_LOGI(TAG, "--- Step 3: Wi-Fi AP ---");
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                                         &event_handler, NULL, NULL));

    wifi_config_t wifi_config = {
        .ap = {
            .ssid           = WIFI_SSID,
            .ssid_len       = strlen(WIFI_SSID),
            .password       = WIFI_PASS,
            .max_connection = 4,
            .authmode       = WIFI_AUTH_WPA2_PSK,
            .channel        = 2,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "AP started: %s | 192.168.4.1", WIFI_SSID);

    /* Main HTTP server (port 80) */
    httpd_handle_t server = NULL;
    httpd_config_t server_config = HTTPD_DEFAULT_CONFIG();
    server_config.stack_size       = 8192;
    server_config.max_uri_handlers = 10;
    server_config.max_open_sockets = 7;

    if (httpd_start(&server, &server_config) == ESP_OK) {
        httpd_uri_t uri_index   = { .uri="/",        .method=HTTP_GET, .handler=index_handler   };
        httpd_uri_t uri_control = { .uri="/control", .method=HTTP_GET, .handler=control_handler };
        httpd_uri_t uri_imu     = { .uri="/imu",     .method=HTTP_GET, .handler=imu_handler     };
        httpd_uri_t uri_cmd     = { .uri="/cmd",     .method=HTTP_GET, .handler=cmd_handler     };
        httpd_uri_t uri_pid     = { .uri="/pid",     .method=HTTP_GET, .handler=pid_handler     };
        httpd_uri_t uri_ws      = { .uri="/ws",      .method=HTTP_GET, .handler=ws_handler,
                                    .is_websocket=true };
        httpd_register_uri_handler(server, &uri_index);
        httpd_register_uri_handler(server, &uri_control);
        httpd_register_uri_handler(server, &uri_imu);
        httpd_register_uri_handler(server, &uri_cmd);
        httpd_register_uri_handler(server, &uri_pid);
        httpd_register_uri_handler(server, &uri_ws);
    }

    /* Stream server (port 81) — separate server to avoid blocking main */
    httpd_handle_t stream_server = NULL;
    httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();
    stream_config.server_port    = 81;
    stream_config.ctrl_port      = 32769;
    stream_config.stack_size     = 8192;
    stream_config.max_uri_handlers = 2;
    stream_config.max_open_sockets = 2;

    if (httpd_start(&stream_server, &stream_config) == ESP_OK) {
        httpd_uri_t uri_stream = { .uri="/stream", .method=HTTP_GET, .handler=stream_handler };
        httpd_register_uri_handler(stream_server, &uri_stream);
    }

    ESP_LOGI(TAG, "Server ready — http://192.168.4.1");
}
