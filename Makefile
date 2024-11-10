.SILENT:

# Define the compiler and compiler flags
NVCC = nvcc
CFLAGS = -L/cnpy -lcnpy -lz --std=c++11

# Define the target executable
TARGET = test

# Define the source file
SRC = data_aug.cu

# Default target
all: $(TARGET)

# Build the target
$(TARGET): $(SRC)
	$(NVCC) -o $(TARGET) $(SRC) $(CFLAGS)

# Run the target and measure time, saving to log file
run: $(TARGET)
	@echo "Running $(TARGET)..."
	@start_time=$$(date +%s) ; \
	./$(TARGET) ; \
	end_time=$$(date +%s) ; \
	duration=$$((end_time - start_time)) ; \
	echo "Time taken: $$duration seconds" > run_log.txt

# Clean up the build
clean:
	@echo "Cleaning up..."
	rm -f $(TARGET)
