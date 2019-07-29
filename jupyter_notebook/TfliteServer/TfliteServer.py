import sys
sys.path.insert(0, './')
sys.path.insert(1, '../')
import flatbuffers
import MnistProt.Mode
import MnistProt.Command
import MnistProt.Commands
import MnistProt.InferenceInput
import MnistProt.InferenceOutput
import socketserver
import numpy as np
import tensorflow as tf
import time
import threading

# Static initializations
interpreter = object()
input_details = []
output_details = []
tfliteLock = threading.Lock()
keep_running = 1

def initInterpreter(model_path):
    """Initializes the tflite interpreter with the given tflite model"""
    global interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    global input_details
    input_details = interpreter.get_input_details()
    global output_details
    output_details = interpreter.get_output_details()

def runInference(data):
    """Runs the tflite model inference

    This function will run the tflite model inference on the given data.
    For this model data are expected to be a (1,28,28,1) numpy tensor.

    :param data: A numpy tensor
    :return: A numpy tensor with the result

    """
    # re-shape digit from (784,) to (1, 28, 28, 1)
    digit = data.reshape(28,28)
    digit = np.expand_dims(digit, axis=0)
    digit = np.expand_dims(digit, axis=3)
    # print(digit.shape)
    interpreter.set_tensor(input_details[0]['index'], digit)
    
    inference_start_time = time.time()
    interpreter.invoke()
    inference_end_time = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    return output_data, (inference_end_time - inference_start_time)*1000.0

class FbTcpHandler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            handle_start_time = time.time()
            # Important: wait for all data!
            data = bytearray()
            while len(data) < 3180:
                data += self.request.recv(1024)
            # print(len(data))

            # parse data
            resp = MnistProt.Commands.Commands.GetRootAsCommands(data, 0)
            # print('Command: %d'% resp.Cmd())
            if resp.Cmd() != MnistProt.Command.Command.CMD_INFERENCE_INPUT:
                print('Unknown protocol command')
                return
            
            # print('Running inference...')
            # print("Input size: %d" % resp.Input().DigitLength())
            tfliteLock.acquire()
            output_data, time_ms = runInference(resp.Input().DigitAsNumpy())
            tfliteLock.release()

            # Build response
            builder = flatbuffers.Builder(128)
            numElems = 10
            MnistProt.InferenceOutput.InferenceOutputStartOutputFVector(builder, numElems)
            for i in reversed(range(0, numElems)):
                builder.PrependFloat32(output_data[i])
            output_vect = builder.EndVector(numElems)
            MnistProt.InferenceOutput.InferenceOutputStart(builder)
            MnistProt.InferenceOutput.InferenceOutputAddTimerMs(builder, time_ms)
            MnistProt.InferenceOutput.InferenceOutputAddOutputN(builder, numElems)
            MnistProt.InferenceOutput.InferenceOutputAddOutputF(builder, output_vect)
            output = MnistProt.InferenceInput.InferenceInputEnd(builder)

            MnistProt.Commands.CommandsStart(builder)
            MnistProt.Commands.CommandsAddCmd(builder, MnistProt.Command.Command.CMD_INFERENCE_OUTPUT)
            MnistProt.Commands.CommandsAddOuput(builder, output)
            req = MnistProt.Commands.CommandsEnd(builder)
            builder.Finish(req)
            buf = builder.Output()

            # Send data
            self.request.sendall(buf)
            # print('Sent %d bytes' % len(buf))

            handle_end_time = time.time()
            print("==== Results ====")
            print('Hander time in msec: %f' % ((handle_end_time - handle_start_time)*1000.0))
            print("Prediction results:", output_data)
            print("Predicted value:", np.argmax(output_data))
            print("\n")

        except Exception as e:
            print(e)
            print('Exception error in TCP. Releasing lock')
            tfliteLock.release()
            self.server.server_close()


class TfliteServer():
    def __init__(self, model_path):
        self._server =  None
        initInterpreter(model_path)
        print('TfliteServer initialized')

    def __del__(self):
        if self._server:
            self._server.server_close()
        
    def close(self):
        if self._server:
            self._server.server_close()

    def listen(self, ip, port):
        print('TCP server started at port: %d' % port)
        self._server = socketserver.TCPServer((ip, port), FbTcpHandler)
        # while keep_running:
        #     self._server.handle_request()
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            print('Shuting down server')
            # self._server.socket.close()
            self._server.server_close()
            self._server.shutdown()
            # self._server.serve_forever()


if __name__=="__main__":
    srv = TfliteServer('../mnist.tflite')
    srv.listen('192.168.0.2', 32001)
    # com = FbComm(uart='/dev/ttyUSB0')
    # com.reqStats()

    # digit = np.load('../digit.txt.npy')
    # com.reqInference(digit)
