CC = g++
CFLAGS = -g -std=c++20 -Wall -Wextra -Iinclude 
TARGET = main
SRCS = main.cpp
OBJS = $(SRCS:.c=.o) 

$(TARGET): $(OBJS)
	@$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CC) $(CFLAGS) -cpp $< -o $@

clean:
	rm -f *.o $(TARGET)

run: $(TARGET)
		@./$(TARGET)
		@rm -f *.o $(TARGET)

#$^ pour avoir le nom de la dependences
#$@ pour avoir le nom de la target
#OBJ = $(SRC:.c = o) recuperer le .c et cree un .o du meme nom
