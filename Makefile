TARGETS = main

all: ${TARGETS}

main: main.cpp
	mpicxx -Wall -Wextra $< -o $@