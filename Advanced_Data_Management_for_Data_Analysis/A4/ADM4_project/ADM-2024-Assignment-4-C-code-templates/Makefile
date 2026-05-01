CC=gcc
CFLAGS=-Wformat=2 -Wformat-signedness -Wextra -Wall -Wpedantic -Werror -pedantic-errors -O3
# NOTE:
# You need to implement both a "branched" and a "predicated" version of your sort algorithm.
# However, you only need to implement one variant of each version,
# i.e., depending on your choice of sort algorithm / implementation,
# only *either* the "in-place" *or* the "out-of-place" variant of each version.
BINARIES=\
	ADM-2023-A4-sort-branched-inplace \
	ADM-2023-A4-sort-predicated-inplace \
	ADM-2023-A4-sort-branched-outofplace \
	ADM-2023-A4-sort-predicated-outofplace

%: %.c
	$(CC) $(CFLAGS) -o $@ $<

.PHONY: all clean

all: $(BINARIES)

clean:
	rm -v $(BINARIES)
