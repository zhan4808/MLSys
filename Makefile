CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
TARGET = mlsys

$(TARGET): solver.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

verify: verify.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# For final submission on Ubuntu: static link
static: solver.cpp
	$(CXX) $(CXXFLAGS) -static -o $(TARGET) $<

clean:
	rm -f $(TARGET) verify output*.json

# Verify all benchmarks
verify-all: $(TARGET) verify
	@for f in benchmarks/mlsys-2026-*.json; do \
		echo "\n=== $$f ==="; \
		./$(TARGET) $$f /tmp/out.json 2>/dev/null; \
		./verify $$f /tmp/out.json; \
	done

# Quick test targets
test1: $(TARGET)
	./$(TARGET) benchmarks/mlsys-2026-1.json output-1.json
	@cat output-1.json

test5: $(TARGET)
	./$(TARGET) benchmarks/mlsys-2026-5.json output-5.json
	@cat output-5.json

test9: $(TARGET)
	./$(TARGET) benchmarks/mlsys-2026-9.json output-9.json
	@cat output-9.json

test-example: $(TARGET)
	./$(TARGET) example_problem.json output-example.json
	@cat output-example.json

test-all: $(TARGET)
	@for f in benchmarks/mlsys-2026-*.json; do \
		echo "=== $$f ==="; \
		./$(TARGET) $$f /dev/null 2>&1 | grep "Total latency"; \
	done

.PHONY: clean test1 test5 test9 test-example test-all static
