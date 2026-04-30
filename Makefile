CC      = g++
CFLAGS  = -g -std=c++20 -O2 -Wall -Wextra -Iinclude
LDFLAGS = -lopenblas
TARGET  = main
SRCS    = main.cpp
OBJS    = $(SRCS:.cpp=.o)

$(TARGET): $(OBJS)
	@$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET)

run: $(TARGET)
	@./$(TARGET)
	@rm -f *.o $(TARGET)

#$^ pour avoir le nom de la dependences
#$@ pour avoir le nom de la target
#OBJ = $(SRC:.c = o) recuperer le .c et cree un .o du meme nom
