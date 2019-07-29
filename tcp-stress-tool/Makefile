IDIR =../esp8266-tf-client
CC=g++
CFLAGS=-std=c++11 -I$(IDIR)

LIBS=-lpthread

tcp-stress-tool: tcp-stress-tool.cpp
	$(CC) -o tcp-stress-tool tcp-stress-tool.cpp $(CFLAGS) $(LIBS)

clean:
	rm tcp-stress-tool
