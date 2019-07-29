MNIST tflite cloud server with ESP8266 and Jetson nano
----

This is _(buzzwords are coming)_ a `MNIST TensorFlow Lite Cloud IoT server/client framework`!

In simple words, it's just an arduino firmware for the ESP8266 which acts as a
TCP client and connects to a TCP server to request an inference on the payload.
The server needs to be able to run python3 and TensorFlow.

> Note: This project derived from this blog post [here](https://www.stupid-projects.com/machine-learning-on-embedded-part-5/)
The whole series starts from [here](https://www.stupid-projects.com/machine-learning-on-embedded-part-1/)

This project was tested on a [Jetson nano](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/) and a workstation (without gpu acceleration).
The server needs to be able to run a python3 script or the included jupyter notepad.

## ESP8266 firmware
To build the esp8266 firmware open the `esp8266-tf-client/esp8266-tf-client.ino`
with Arduino IDE (version > 1.8). Then in the file you need to change a couple
of variables according to your network setup. In the source code you'll find
those values:

```cpp
#define SSID "SSID"
#define SSID_PASSWD "PASSWORD"
#define SERVER_IP "192.168.0.123"
#define SERVER_PORT 32001
```

You need to edit them according to your network. So, use your wifi router's SSID and
password. The `SERVER_IP` is the IP of the computer that will run the python
server and the `SERVER_PORT` is the server's port and they both need to be the same
in the python script.

All the data in the communication between the client and the server are serialized by
using flatbuffers. This comes with quite a significant performance hit but it's quite
necessary in this case. The client sends 3180 bytes on every transaction to the server,
which are the serialized 784 floats for each 28x28 byte digit. Then the response from
the server to the client is 96 bytes.

By default this project assumes that the esp8266 runs at 160MHz. In case you change
this to 80MHz then you need also to change the `MS_CONST` in the code like this:

```cpp
#define MS_CONST 80000.0f
```

Otherwise the ms values will be wrong. I guess there's an easier and automated way
to do this, but I didn't spent much time to find it.

#### Build the firmware
To build the firmware just open the `esp8266-tf-client.ino` with Arduino IDE
and then press `Verify` and `Upload`.

> Note: I've used version 1.8.9 of the IDE and the esp8266 comunity board support
package and version 2.5.2

#### Supported commands
The firmware supports 3 serial commands that you can send via the terminal. All the
commands need to be terminated with a newline. The supported commands are:

* ```TEST```

This command will send a single digit inference request to the server and it
will print the parsed response. This is an example response:

```sh
Request a single inference...
======== Results ========
Inference time in ms: 8.420229
out[0]: 0.080897
out[1]: 0.128900
out[2]: 0.112090
out[3]: 0.129278
out[4]: 0.079890
out[5]: 0.106956
out[6]: 0.074446
out[7]: 0.106730
out[8]: 0.103112
out[9]: 0.077702
Transaction time: 36.726883 ms
```

The `inference time in ms` is the time that the server spend to run the inference. This
time is in ms, so if you get a value like this (0.152) that means that the server needed
152 microseconds. Note that there is a limit on the minimum time difference that any
system can measure, so I guess if your system is very fast you might get a zero value.

Next the `out[x]` values are the inference output. Because now we're using random data,
don't expect them to have any real meaning.

The `transaction time` is quite important and it's the time that the whole transaction
lasted. That includes the data serialization and the TCP transaction. As you can see
from the above example the inference time in the server was 8.42 ms and the whole time
that the esp8266 spent was 36.72 ms, so the data serialization and the TCP send/recv
lasted 28.3 ms.

* ```START=<SECS>```

This command will trigger a TCP inference request from the server every `<SECS>`. Therefore,
if you want to poll the server every 5 secs then you need to send this command over the
serial to the esp8266 (don't forget the newline in the end).

```sh
START=5
```

This command will start a timer which every 5 secs will request an inference. This is part
of the output.

```sh
======== Results ========
Inference time in ms: 6.369591
out[0]: 0.080897
out[1]: 0.128900
out[2]: 0.112090
out[3]: 0.129278
out[4]: 0.079890
out[5]: 0.106956
out[6]: 0.074446
out[7]: 0.106730
out[8]: 0.103112
out[9]: 0.077702
Transaction time: 33.914593 ms
```

As you can see the output is pretty much the same as before.

> Note: The `transaction time` will vary more that the `inference time`. The reason is that
because of the WiFi network and all the delays between the TCP stack of the esp8266 and
the Jetson nano and the network traffic and latency this time can never be without variations.

* ```STOP```

This command will stop the timer that sends the periodical TCP inference requests.

#### Comm protocol
The client sends 3180 bytes on every transaction to the server, which are the
serialized 784 floats for each 28x28 byte digit. Then the response from the
server to the client is 96 bytes. These byte lengths are hardcoded, so if you
do any changes you need also to change he definitions in the code. They are
hard-coded in order to accelerate the network recv() routines so they don't
wait for timeouts.

## Server side
In my case I've tested the python server on a Jetson nano and on my workstation.
My workstation doesn't provide any gpu acceleration for tensorflow and these are
the specs:

#### Workstation
* Ryzen 2700x @ 3700MHz (8 cores / 16 threads)
* 32GB @ 3200MHz
* GeForce GT 710
* Ubuntu 18.04
* Kernel 4.18.20-041820-generic

#### Jetson-nano
* Quad-core ARM Cortex-A57
* 4GB LPDDR4
* NVIDIA Maxwell GPU with 128 CUDA cores
* Ubuntu 18.01
* Kernen 4.9.140-tegra #1 SMP PREEMPT

## Jupyter notebook
To run the jupyter notebook on the Jetson nano follow the instructions in the
notebook. If you have all the proper dependencies installed then run:
```sh
sudo jupyter notebook --allow-root --ip 192.168.0.86 --port 8888
```

Just replace the `ip` and `port` with the proper ones, though.

Then you can run all the cells in the notebook. The last cell will run the
`TfliteServer`, which accepts TCP connections from the esp8266 clients.

## TfliteServer
This TCP server is implemented in python and it's located in
`jupyter_notebook/TfliteServer/TfliteServer.py`. You can run this server on
your terminal like this:
```sh
python3 TfliteServer
```

> Note: Before running the server, you need to change the ip and the port
so they are proper for your netork setup.

#### Benchmarking the TfliteServer
If you want to benchmark the server, then I've written a tool called `tcp-stress-tool`
and it's located in `tcp-stress-tool/tcp-stress-tool.cpp`. To build the tool
run this command:
```sh
make
```

Then you can use the tool like this:
```sh
./tcp-stress-tool [server ip] [server port] [num of connections]
```

For example, if the Jetson nano has the IP 192.168.0.86 and you're running the
script from your desktop, then if the `TfliteServer` server listens on the port
32001 and you need to test with 10 connections (=10 threads), then run this:
```sh
./tcp-stress-tool 192.168.0.86 32001 10
```

I've found out that the Jetson nano can handle more that 10-20 simultaneous TCP
connections, though my desktop (2700x) could handle easily more than 500. I think
though, that this probably an issue with the python sockets rather the hardware,
but I'm not sure. Maybe in the future I'll implement the server with C++ and
re-test.

