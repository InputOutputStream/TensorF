# ══════════════════════════════════════════════════════════════════════════════
#  TensorF — Master Makefile
#  Structure attendue:
#    ./ (racine)           ← CPP sources + ce Makefile
#    ./include/            ← tous les headers
#    ./bin/                ← executables produits (créé automatiquement)
# ══════════════════════════════════════════════════════════════════════════════

CXX   := g++
STD   := -std=c++20
OPT   := -O2
WARN  := -Wall -Wextra -Wno-unused-parameter
DEBUG ?= 0

ifeq ($(DEBUG),1)
    OPT  := -O0 -g3 -fsanitize=address,undefined
    LSAN := -fsanitize=address,undefined
else
    LSAN :=
endif

ROOT := .
BIN  := $(ROOT)/bin

# ── Include paths ─────────────────────────────────────────────────────────────

INC := \
    -I$(ROOT) \
    -I$(ROOT)/include \
    -I$(ROOT)/include/core \
    -I$(ROOT)/include/nn \
    -I$(ROOT)/include/data \
    -I$(ROOT)/include/net \
    -I$(ROOT)/include/tools

# ── Librairies système ────────────────────────────────────────────────────────
LIBS     := -lpthread

# OpenBLAS — cherche d'abord via pkg-config, sinon fallback -lblas
OPENBLAS := $(shell pkg-config --libs openblas 2>/dev/null)
ifeq ($(OPENBLAS),)
    LIBS += -lblas
else
    LIBS += $(OPENBLAS)
endif

# nlohmann/json — header-only, juste besoin du -I si pas dans /usr/include
JSON_INC := $(shell pkg-config --cflags nlohmann_json 2>/dev/null)
INC      += $(JSON_INC)

CXXFLAGS := $(STD) $(OPT) $(WARN) $(INC)
LDFLAGS  := $(LIBS) $(LSAN)

$(shell mkdir -p $(BIN))

# ══════════════════════════════════════════════════════════════════════════════
.PHONY: all gpt2 llama smollm transformer server client benchmark tests \
        clean install-deps check-deps help

all: check-deps gpt2 smollm transformer benchmark server client tests


# ── GPT-GPT-2 inference ───────────────────────────────────────────────────────────
gpt2: $(BIN)/gpt2
$(BIN)/gpt2: examples/GPT2.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# ── LLAMA-SmolLM2 inference ─────────────────────────────────────────────────────────
smollm: $(BIN)/smollm
$(BIN)/smollm: examples/SmollLLM.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# ── Character-level Llama training ───────────────────────────────────────────
transformer: $(BIN)/transformer
$(BIN)/transformer: examples/transformer.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# ── Federated learning — server ───────────────────────────────────────────────

SERVER_SRC ?= include/net/Network/Server.cpp
CLIENT_SRC ?= include/net/Network/Client.cpp

server: $(BIN)/server
$(BIN)/server: $(SERVER_SRC)
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

client: $(BIN)/client
$(BIN)/client: $(CLIENT_SRC)
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# ── Benchmark / profiler ──────────────────────────────────────────────────────
benchmark: $(BIN)/benchmark
$(BIN)/benchmark: tests/benchmark.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# ── Tests unitaires ───────────────────────────────────────────────────────────
tests: $(BIN)/basic_tests
$(BIN)/basic_tests: tests/basic_tests.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/basic_tests"
	$(BIN)/basic_tests

# ── Network  ───────────────────────────────────────────────────────────

net: client server

# ── Network  ───────────────────────────────────────────────────────────

examples: gpt2 smollm transformer

# ── Nettoyage ─────────────────────────────────────────────────────────────────
clean:
	@echo "[CLEAN] $(BIN)/"
	rm -rf $(BIN)

# ══════════════════════════════════════════════════════════════════════════════
#  Installation des dépendances (Ubuntu / Debian)
# ══════════════════════════════════════════════════════════════════════════════
install-deps:
	sudo apt-get update -qq
	sudo apt-get install -y --no-install-recommends \
	    build-essential g++ cmake pkg-config \
	    libopenblas-dev nlohmann-json3-dev libxxhash-dev \
	    wget curl

check-deps:
	@echo "── Compilateur"
	@$(CXX) --version | head -1
	@echo "── nlohmann/json"
	@(pkg-config --modversion nlohmann_json 2>/dev/null && echo "  OK pkg-config") \
	 || (test -f /usr/include/nlohmann/json.hpp && echo "  OK header") \
	 || echo "  MANQUANT — sudo apt install nlohmann-json3-dev"
	@echo "── OpenBLAS"
	@pkg-config --modversion openblas 2>/dev/null && echo "  OK" \
	 || echo "  MANQUANT — sudo apt install libopenblas-dev"
	@echo "── pthreads"
	@echo "#include <pthread.h>" | $(CXX) -x c++ - -lpthread -o /dev/null 2>/dev/null \
	 && echo "  OK" || echo "  MANQUANT"

# ══════════════════════════════════════════════════════════════════════════════
help:
	@echo ""
	@echo "  make [all]          gpt2 + llama + smollm + transformer + benchmark"
	@echo "  make gpt2           GPT-2 inference"
	@echo "  make llama          Llama (SmolLM2) inference"
	@echo "  make smollm         SmolLM2 complet (avec dataset)"
	@echo "  make transformer    Entraînement character-level"
	@echo "  make server         Serveur federated (besoin de include/net/Network/server.cpp)"
	@echo "  make client         Client federated  (besoin de include/net/Network/client.cpp)"
	@echo "  make benchmark      Profiler matériel"
	@echo "  make tests          Tests unitaires"
	@echo "  make clean          Supprime bin/"
	@echo "  make install-deps   Installe les libs système"
	@echo "  make check-deps     Vérifie les libs"
	@echo ""
	@echo "  DEBUG=1 make <cible>   ASan + UBSan + pas d'optimisation"
	@echo "  SERVER_SRC=mon_serveur.cpp make server"
	@echo ""