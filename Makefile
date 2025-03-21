# Makefile
JULIA ?= julia

ROOT_DIR:=$(shell pwd)
TARGET="CAMNASCompiled"

.PHONY: all, default, clean, new

default: all

all: camnasjl.so wrapper.o plugin.so

new: clean all

clean:
	rm -rf $(ROOT_DIR)/$(TARGET)
	rm -f $(ROOT_DIR)/camnasjl.so

camnasjl.so: CAMNAS/build/build.jl CAMNAS/src/CAMNAS.jl CAMNAS/src/mna_solver.jl CAMNAS/src/config.jl CAMNAS/build/precompile_statements.jl
	$(JULIA) --project=CAMNAS --threads=auto --startup-file=no CAMNAS/build/build.jl $(TARGET)

wrapper.o:
	gcc -c -fPIC -O2 -I$(ROOT_DIR)/CAMNASCompiled/include -I$(ROOT_DIR)/../../../include dpsim_wrapper.c

plugin.so: wrapper.o
	gcc -shared -o camnasjl.so dpsim_wrapper.o  -L$(ROOT_DIR)/CAMNASCompiled/lib -lcamnasjl
