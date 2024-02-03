CC=g++
CFLAGS=-I/usr/local/include -L/usr/local/lib -lopencv_core -lopencv_highgui -std=c++11

HELLODEPS=opencv-hello.cpp

hello: $(HELLODEPS)
	$(CC) $(CFLAGS) $(HELLODEPS) -o bin/hello

