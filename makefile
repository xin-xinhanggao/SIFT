sift: main.o SiftHelper.o SIFT.o
	g++ $$(pkg-config --cflags --libs opencv) main.o SiftHelper.o SIFT.o -o sift

main.o: main.cpp SiftHelper.hpp
	g++ $$(pkg-config --cflags --libs opencv) -c main.cpp

SiftHelper.o: SiftHelper.hpp SiftHelper.cpp SIFT.hpp
	g++ $$(pkg-config --cflags --libs opencv) -c SiftHelper.cpp

SIFT.o: SIFT.hpp __SIFT.hpp SIFT.cpp
	g++ $$(pkg-config --cflags --libs opencv) -c SIFT.cpp
	
clean: 
	rm *.o
