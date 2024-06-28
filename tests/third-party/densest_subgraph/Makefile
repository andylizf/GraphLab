CXX = g++
CXXFLAGS = -Wall -g

SRCS = densest_subgraph.cpp
EXEC = densest_subgraph

all: $(EXEC)

$(EXEC): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(EXEC)

.PHONY: all clean
