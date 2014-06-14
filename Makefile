CC = g++
CFLAGS = -O0 -g -w -Wall -Wno-deprecated -L.
OBJ  = ./main.o ./m3n.o ./fun.o
LINKOBJ  = ./main.o ./m3n.o ./fun.o
BIN  = ./m3n-ext


LIBRARIES = -lm -lnsl -lstdc++ -lgzstream -lz

.PHONY: all all-before all-after clean clean-custom

all: $(BIN)


clean: clean-custom
		rm -f $(OBJ) $(BIN)

$(BIN): $(LINKOBJ)
		$(CC) $(CFLAGS) $(LINKOBJ) -o $(BIN) $(LIBRARIES)
		cp -f $(BIN) $(HOME)/bin

./main.o: main.cpp m3n.h fun.h
		$(CC) $(CFLAGS) -c ./main.cpp -o ./main.o

./m3n.o: m3n.cpp m3n.h fun.h
		$(CC) $(CFLAGS) -c m3n.cpp -o m3n.o

./fun.o: fun.cpp fun.h
		$(CC) $(CFLAGS) -c fun.cpp -o fun.o
		