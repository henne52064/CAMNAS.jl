# Makefile
JULIA ?= julia

ROOT_DIR:=$(shell pwd)
TARGET="JuliaMNACompiled"

.PHONY: all, default, clean, new

default: all

all: juliamna.so wrapper.o plugin.so

new: clean all

clean:
	rm -rf $(ROOT_DIR)/$(TARGET)
	rm -f $(ROOT_DIR)/juliamna.so

juliamna.so: JuliaMNA/build/build.jl JuliaMNA/src/JuliaMNA.jl JuliaMNA/src/mna_solver.jl JuliaMNA/src/config.jl JuliaMNA/build/precompile_statements.jl
	$(JULIA) --project=JuliaMNA --threads=auto --startup-file=no JuliaMNA/build/build.jl $(TARGET)

wrapper.o:
	gcc -c -fPIC -O2 -I$(ROOT_DIR)/JuliaMNACompiled/include -I$(ROOT_DIR)/../../../include dpsim_wrapper.c

plugin.so: wrapper.o
	gcc -shared -o juliamna.so dpsim_wrapper.o  -L$(ROOT_DIR)/JuliaMNACompiled/lib -ljuliamna
