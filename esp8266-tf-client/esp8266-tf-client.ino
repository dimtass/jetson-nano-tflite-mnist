/**
 * MNIST TF-Lite cloud client
 * 
 * This is an ESP8266 TCP client that is able to connect to a cloud
 * tflite server and request from the server to run an inference on
 * the input (random in this case)
 * 
 * Copyright 2019 Dimitris Tassopoulos <dimtass@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mnist_schema_generated.h"
#include <ESP8266WiFi.h>
#include <ESP8266WiFiMulti.h>
#include <Ticker.h> //Ticker Library

ESP8266WiFiMulti WiFiMulti;

#define SSID "SSID"
#define SSID_PASSWD "PASSWORD"
#define SERVER_IP "192.168.0.86"

#define SERVER_PORT 32001
#define MNIST_DIGIT_SIZE 784
#define MNIST_RESP_SIZE 96

#define MS_CONST 160000.0f

const char *ssid = SSID;
const char *ssid_passwd = SSID_PASSWD;
const char *server_ip = SERVER_IP;
const uint16_t server_port = SERVER_PORT;
String cmd = "";
Ticker polling_timer;
volatile bool polling_triggered = false;

/* Always allocate big buffers in heap */
float digit[MNIST_DIGIT_SIZE];
flatbuffers::FlatBufferBuilder fbb(1024);

static inline unsigned get_ccount(void)
{
    unsigned r;
    asm volatile("rsr %0, ccount" : "=r"(r));
    return r;
}

void setup()
{
    Serial.begin(115200);

    printf("Program started...\n");
    printf("Trying to connect to %s\n", ssid);

    WiFi.mode(WIFI_STA);
    WiFiMulti.addAP(ssid, ssid_passwd);
}

void ICACHE_RAM_ATTR trigger_polling()
{
    polling_triggered = true;
}

void ICACHE_RAM_ATTR CreateFbCmd()
{
    unsigned tick_start = get_ccount(); // start timer

    /* Create buffer to send */
    fbb.Clear(); // clear any previous data

    auto out_vect = fbb.CreateVector((float *)digit, MNIST_DIGIT_SIZE);
    auto input_f = MnistProt::CreateInferenceInput(fbb, out_vect);

    MnistProt::CommandsBuilder builder(fbb);
    builder.add_cmd(MnistProt::Command_CMD_INFERENCE_INPUT);
    builder.add_input(input_f);
    auto resp = builder.Finish();
    fbb.Finish(resp);

    uint8_t *buf = fbb.GetBufferPointer();
    int buf_size = fbb.GetSize();

    // printf("Buffer size: %d\n", buf_size);

    /* Connect to server and send data */
    WiFiClient client;

    if (!client.connect(server_ip, server_port)) {
        Serial.println("Failed to connect to server");
        delay(1000);
        return;
    }
    /* send flatbuffer */
    client.write(buf, buf_size);
    /* recv data */
    uint8_t recv_buf[MNIST_RESP_SIZE];
    /* wait for data */
    int recv_len = client.readBytes(recv_buf, MNIST_RESP_SIZE);
    client.stop();

    unsigned tick_end = get_ccount(); // stop timer
    // printf("Received %d bytes\n", recv_len);

    if (recv_len == MNIST_RESP_SIZE) {
        /* parse data */
        auto req = MnistProt::GetCommands(recv_buf);
        flatbuffers::Verifier verifier(reinterpret_cast<unsigned char *>(recv_buf), recv_len);
        bool isCommand = req->Verify(verifier);
        if (!isCommand) {
            printf("Invalid flatbuffer data received\n");
            return;
        }
        if (req->cmd() == MnistProt::Command_CMD_INFERENCE_OUTPUT) {
            auto resp = req->ouput();

            printf("======== Results ========\n");
            printf("Inference time in ms: %f\n", resp->timer_ms());
            float *p = (float *)resp->output_f()->data();
            for (int i = 0; i < resp->output_n(); i++) {
                printf("out[%d]: %f\n", i, p[i]);
            }
        }
    }
    printf("Transaction time: %f ms\n\n", ((float)(tick_end - tick_start)) / MS_CONST);
}

void loop()
{
    // wait for WiFi connection
    if ((WiFiMulti.run() == WL_CONNECTED)) {

        if (Serial.available()) {
            char tmp = Serial.read();
            cmd.concat(tmp);
        }

        if (cmd != "" && (cmd.indexOf('\n') >= 0)) {
            if (cmd.indexOf("TEST") >= 0) {
                // Get mode
                cmd = "";
                printf("Request a single inference...\n");
                CreateFbCmd();
            } else if (cmd.indexOf("START=") >= 0) {
                // Get mode
                char c_secs[3] = {0};
                c_secs[0] = cmd[6];
                int secs = atoi(c_secs);
                cmd = "";
                printf("Polling server every %d secs\n", secs);
                polling_timer.attach(secs, trigger_polling);

            } else if (cmd.indexOf("STOP") >= 0) {
                Serial.println("Stoping polling mode...");
                polling_timer.detach();
                cmd = "";
            }
        }

        if (polling_triggered) {
            polling_triggered = false;
            CreateFbCmd();
        }
    }
}
