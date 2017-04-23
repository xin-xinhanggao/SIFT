sift: main.o match.o SIFT.o
	g++ $$(pkg-config --cflags --libs opencv) main.o match.o SIFT.o -o sift

main.o: main.cpp match.hpp
	g++ $$(pkg-config --cflags --libs opencv) -c main.cpp

match.o: match.hpp match.cpp SIFT.hpp
	g++ $$(pkg-config --cflags --libs opencv) -c match.cpp

SIFT.o: SIFT.hpp siftfunction.hpp SIFT.cpp
	g++ $$(pkg-config --cflags --libs opencv) -c SIFT.cpp
	
clean: 
	rm *.o
