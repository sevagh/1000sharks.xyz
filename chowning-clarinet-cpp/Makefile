LD_LIBRARY_PATH:=$(LD_LIBRARY_PATH):/usr/local/lib
CXX_FLAGS:=$(CXX_FLAGS) -I$(STK_INCLUDE_DIR) -lpthread -lstk -Wl,-rpath /usr/local/lib

all: build

fmt:
	astyle -A8 -i stkclarinet.cpp
build:
	$(CXX) $(CXX_FLAGS) stkclarinet.cpp -o stkclarinet

clean:
	rm stkclarinet

.PHONY: clean fmt
