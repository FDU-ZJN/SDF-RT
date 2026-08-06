CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -Werror -std=c11

.PHONY: all clean

all: sdf_uram_loader

sdf_uram_loader: sdf_uram_loader.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f sdf_uram_loader
