TARGETS = main one-sided

all: ${TARGETS}

.PHONY: clean
clean:
	rm -f ${TARGETS} *.o

MPICXX := mpicxx
CXXFLAGS := -std=c++11 -Wall -Wextra -Wshadow -pedantic -O2

main: main.cpp Makefile
	$(MPICXX) $(CXXFLAGS) $< -o $@

one-sided: one_sided.cpp Makefile
	$(MPICXX) $(CXXFLAGS) $< -o $@