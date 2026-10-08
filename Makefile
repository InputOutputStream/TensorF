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

# OpenBLAS — d'abord via pkg-config, sinon fallback -lblas
OPENBLAS := $(shell pkg-config --libs openblas 2>/dev/null)
ifeq ($(OPENBLAS),)
    LIBS += -lblas
else
    LIBS += $(OPENBLAS)
endif

# nlohmann/json — header-only
JSON_INC := $(shell pkg-config --cflags nlohmann_json 2>/dev/null)
INC      += $(JSON_INC)

CXXFLAGS := $(STD) $(OPT) $(WARN) $(INC)
LDFLAGS  := $(LIBS) $(LSAN)

$(shell mkdir -p $(BIN))

# ══════════════════════════════════════════════════════════════════════════════
.PHONY: all gpt2 llama smollm transformer benchmark tests \
        gateway run-gateway clean install-deps check-deps help # server client

all: check-deps gpt2 smollm transformer benchmark tests gateway # server client


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

SERVER_SRC ?= tests/GPT/Server.cpp
CLIENT_SRC ?= tests/GPT/Client.cpp

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
# 	$(BIN)/basic_tests

# ── GNN Tests ───────────────────────────────────────────────────────────

# ── gcntest  ───────────────────────────────────────────────────────────
tests: $(BIN)/gcntest
$(BIN)/gcntest: tests/gcntest.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/gcntest"
# 	$(BIN)/gcntest

# ── spmmtest  ───────────────────────────────────────────────────────────

tests: $(BIN)/spmmtest
$(BIN)/spmmtest: tests/spmmtest.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/spmmtest"
# 	$(BIN)/spmmtest

# ── coratrain  ───────────────────────────────────────────────────────────

tests: $(BIN)/coratrain
$(BIN)/coratrain: tests/coratrain.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/coratrain"
# 	$(BIN)/coratrain

# ── contract  ───────────────────────────────────────────────────────────

tests: $(BIN)/contract
$(BIN)/contract: tests/contract.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/contract"
# 	$(BIN)/contract

# ── diff_ops  ───────────────────────────────────────────────────────────

tests: $(BIN)/diff_ops
$(BIN)/diff_ops: tests/diff_ops.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/diff_ops"
# 	$(BIN)/diff_ops

# ── matrix_tests  ───────────────────────────────────────────────────────────

tests: $(BIN)/matrix_tests
$(BIN)/matrix_tests: tests/matrix_tests.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/matrix_tests"
# 	$(BIN)/matrix_tests

# ── scope_test  ───────────────────────────────────────────────────────────

tests: $(BIN)/scope_test
$(BIN)/scope_test: tests/scope_test.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/scope_test"
# 	$(BIN)/scope_test

# ── support_tests  ───────────────────────────────────────────────────────────

tests: $(BIN)/support_tests
$(BIN)/support_tests: tests/support_tests.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/support_tests"
# 	$(BIN)/support_tests

# ── Network  ───────────────────────────────────────────────────────────

net: client server

# ── Web dashboard gateway (REST + WebSocket) ──────────────────────────────────
GATEWAY_SRC := gateway/src/main.cpp
GATEWAY_INC := -Igateway/include

gateway: $(BIN)/tensorf-gateway
$(BIN)/tensorf-gateway: $(GATEWAY_SRC) gateway/include/JobManager.hpp gateway/include/WsHub.hpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $(GATEWAY_INC) $< -o $@ $(LDFLAGS)

# Run the gateway. Override PORT / TENSORF_BIN_DIR as needed:
#   PORT=9090 make run-gateway
PORT ?= 8080
TENSORF_BIN_DIR ?= $(abspath $(BIN))

run-gateway: gateway
	TENSORF_BIN_DIR=$(TENSORF_BIN_DIR) $(BIN)/tensorf-gateway --port $(PORT)

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
	
	@echo "int main(){}" | $(CXX) -x c++ - -pthread -o /dev/null && echo OK || echo MANQUANT
# ══════════════════════════════════════════════════════════════════════════════
help:
	@echo ""
	@echo "  make [all]          gpt2 + llama + smollm + transformer + benchmark"
	@echo "  make gpt2           GPT-2 inference"
	@echo "  make llama          Llama (SmolLM2) inference"
	@echo "  make smollm         SmolLM2 complet (avec dataset)"
	@echo "  make transformer    Entraînement character-level"
# 	@echo "  make server         Serveur federated (besoin de tests/GPT/server.cpp)"
# 	@echo "  make client         Client federated  (besoin de tests/GPT/client.cpp)"
	@echo "  make benchmark      Profiler matériel"
	@echo "  make tests          Tests unitaires"
	@echo "  make gcntest        Test du gcn"
	@echo "  make gateway        Web dashboard gateway (REST + WebSocket, gateway/src/main.cpp)"
	@echo "  make run-gateway    Build + lance le gateway (PORT=8080 par défaut)"
	@echo "  make clean          Supprime bin/"
	@echo "  make install-deps   Installe les libs système"
	@echo "  make check-deps     Vérifie les libs"
	@echo ""
	@echo "  DEBUG=1 make <cible>   ASan + UBSan + pas d'optimisation"
	@echo "  SERVER_SRC=mon_serveur.cpp make server"
	@echo ""