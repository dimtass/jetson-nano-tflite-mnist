 /** 
  * tcp-stress-tool
  * 
  * This tool is part of this project here:
  * https://www.stupid-projects.com/machine-learning-on-embedded-part-5/
  * https://bitbucket.org/dimtass/jetson-nano-tflite-mnist
  * 
  * I've written this in order to stress a python TCP server that
  * runs tflite inferences.
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

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "mnist_schema_generated.h"

#define MAX_CLIENTS 4096
#define MAX_SIZE 3180
#define MNIST_DIGIT_SIZE 784
#define MNIST_RESP_SIZE 96

float * time_results;
pthread_mutex_t lock;

char *server_ip = NULL;
int server_port = 0;
int clients_num = 0;

void *connection_handler(void *threadid);

double get_time()
{
    struct timeval t;
    struct timezone tzp;
    gettimeofday(&t, &tzp);
    return t.tv_sec + t.tv_usec*1e-6;
}

int main(int argc, char **argv)
{
    char tmp[100];

    int aflag = 0;
    int bflag = 0;

    printf("This tool will spawn a number of TCP clients and will request\n"
            "the tflite server to run an inference on random data.\n"
            "Warning: there is no proper input parsing, so you need to be\n"
            "cautious and read the usage below.\n\n"
            "Usage:\n"
            "tcp-stress-tool [server ip] [server port] [number of clients]\n\n");

    server_ip = argv[1];
    server_port = atoi(argv[2]);
    clients_num = atoi(argv[3]);


    printf("Using:\n"
            "server ip: %s\n"
            "server port: %d\n"
            "number of clients: %d\n\n",
            server_ip, server_port, clients_num);

    if (clients_num <= 0) {
        printf("The number of clients needs to be a number > 0, got: %d\n", clients_num);
        return 1;
    }
    if ((server_port == 0) || (server_port > 0xFFFF)) {
        printf("The server port needs to be >0 and <%d\n", 0xFFFF);
        return 1;
    }

    printf("Spawning %d TCP clients...\n", clients_num);

    /* Create an array of floats to save the time results */
    time_results = (float*) malloc(sizeof(float) * clients_num);

    int socket_desc , new_socket , c , *new_sock, i;

    pthread_t tflite_client_thread[clients_num];

    double startTime = get_time();

    for (i=1; i<=clients_num; i++) {
        if( pthread_create( &tflite_client_thread[i], NULL ,  connection_handler , (void*) i) < 0)
        {
            printf("[%d] could not create thread\n", i);
        }
    }
    /* Now wait for all threads */
    for (i=1; i<=clients_num; i++) {
       pthread_join(tflite_client_thread[i], NULL);
    }

    double elapsedTime = get_time()-startTime;
    printf("\n----------------------\n");
    printf("Total elapsed time: %f ms\n", elapsedTime*1000);

    float avg_server_time = 0.0f;
    for (int i=0; i<clients_num; i++) {
        avg_server_time += time_results[i];
    }
    avg_server_time = avg_server_time / clients_num;
    printf("Average server inference time: %f ms\n", avg_server_time);

    free(time_results);

    return 0;
}

void *connection_handler(void *threadid)
{
    int threadnum = (intptr_t)threadid;
    int sock_desc;
    struct sockaddr_in serv_addr;
    uint8_t recv_buf[MNIST_RESP_SIZE];

    /* prepare data */
    float digit[MNIST_DIGIT_SIZE];
    flatbuffers::FlatBufferBuilder fbb(MAX_SIZE);

    auto out_vect = fbb.CreateVector((float *)digit, MNIST_DIGIT_SIZE);
    auto input_f = MnistProt::CreateInferenceInput(fbb, out_vect);

    MnistProt::CommandsBuilder builder(fbb);
    builder.add_cmd(MnistProt::Command_CMD_INFERENCE_INPUT);
    builder.add_input(input_f);
    auto resp = builder.Finish();
    fbb.Finish(resp);

    uint8_t *buf = fbb.GetBufferPointer();
    int buf_size = fbb.GetSize();

    /* Connect to server */
    if((sock_desc = socket(AF_INET, SOCK_STREAM, 0)) < 0)
        printf("[thread=%d] Failed creating socket.\n", threadnum);

    bzero((char *) &serv_addr, sizeof (serv_addr));

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(server_ip);
    serv_addr.sin_port = htons(server_port);

    struct timeval tv;
    tv.tv_sec = 30;        // 30 Secs Timeout
    tv.tv_usec = 0;        // Not init'ing this can cause strange errors
    setsockopt(sock_desc, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv,sizeof(struct timeval));

    /* Connect to the server */
    if (connect(sock_desc, (struct sockaddr *) &serv_addr, sizeof (serv_addr)) < 0) {
        printf("[thread=%d] Failed to connect to server\n", threadnum);
    }

    printf("[thread=%d] Connected\n", threadnum);
 
    send(sock_desc, buf, buf_size, 0);

    int recv_len = recv(sock_desc, recv_buf, MNIST_RESP_SIZE, 0);
    if (!recv_len) {
        printf("[thread=%d] No response?", threadnum);
        goto _clean;
    }

    if (recv_len == MNIST_RESP_SIZE) {
        /* parse data */
        auto req = MnistProt::GetCommands(recv_buf);
        flatbuffers::Verifier verifier(reinterpret_cast<unsigned char *>(recv_buf), recv_len);
        bool isCommand = req->Verify(verifier);
        if (!isCommand) {
            printf("[thread=%d] Invalid flatbuffer data received\n", threadnum);
            goto _clean;
        }
        if (req->cmd() == MnistProt::Command_CMD_INFERENCE_OUTPUT) {
            auto resp = req->ouput();
            pthread_mutex_lock(&lock);
            time_results[threadnum] = resp->timer_ms();
            pthread_mutex_unlock(&lock);
        }
    }
    printf("[thread=%d]  Inference time in ms: %f\n", threadnum, time_results[threadnum]);

_clean:
    bzero(recv_buf, MNIST_RESP_SIZE);
    close(sock_desc);
    return 0;
}