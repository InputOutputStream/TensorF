# ══════════════════════════════════════════════════════════════════════════════
#  TensorF — Master Makefile
#  Structure attendue:
#    ./ (racine)           ← CPP sources + ce Makefile
#    ./include/            ← tous les headers
#    ./bin/                ← executables produits (créé automatiquement)
# ══════════════════════════════════════════════════════════════════════════════

CXX   := g++
STD   := -std=c++20
# ARCH: -march value. native = fastest on THIS machine (binaries will not run on older CPUs). ARCH= (empty) = portable build.
# Cross-compiling for Android? Use the NDK toolchain (CXX=aarch64-linux-android24-clang++ ARCH=) rather than this g++.
ARCH  ?= native
OPT   := -O2 $(if $(ARCH),-march=$(ARCH))
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
    -I$(ROOT)/include/profiler \
    -I$(ROOT)/include/parallel \
    -I$(ROOT)/include/tools

# ── Librairies système ────────────────────────────────────────────────────────
LIBS     := -lpthread

# ── BLAS ─────────────────────────────────────────────────────────────────────
#   make BLAS=auto       (défaut) OpenBLAS si trouvé, sinon le shim portable
#   make BLAS=openblas   OpenBLAS obligatoire (erreur s'il est absent)
#   make BLAS=shim       boucles portables de include/tools/cblas_shim.h (lent, aucune lib)
#   make BLAS=none       Matrix sans BLAS (boucles internes)
#   OPENBLAS_ROOT=/opt/OpenBLAS   OpenBLAS hors des chemins système (include/ et lib/)
# Le choix est transmis au code par -DTENSORF_USE_OPENBLAS (voir include/tools/tensorf_blas.h).
# Plus de repli sur -lblas : c'est le BLAS de référence, plus lent que le shim.
BLAS ?= auto
ifneq ($(OPENBLAS_ROOT),)
    OPENBLAS_CFLAGS := -I$(OPENBLAS_ROOT)/include
    OPENBLAS_LIBS   := -L$(OPENBLAS_ROOT)/lib -Wl,-rpath,$(OPENBLAS_ROOT)/lib -lopenblas
else
    OPENBLAS_CFLAGS := $(shell pkg-config --cflags openblas 2>/dev/null)
    OPENBLAS_LIBS   := $(shell pkg-config --libs openblas 2>/dev/null)
endif
ifeq ($(BLAS),auto)
    ifneq ($(OPENBLAS_LIBS),)
        BLAS := openblas
    else
        BLAS := shim
        $(warning OpenBLAS introuvable (pkg-config) : BLAS=shim, GEMM lent. sudo apt install libopenblas-dev)
    endif
endif
ifeq ($(BLAS),openblas)
    ifeq ($(OPENBLAS_LIBS),)
        $(error BLAS=openblas mais OpenBLAS est introuvable. sudo apt install libopenblas-dev, ou OPENBLAS_ROOT=...)
    endif
    BLAS_INC    := $(OPENBLAS_CFLAGS)
    BLAS_DEFS   := -DTENSORF_USE_OPENBLAS
    LIBS        += $(OPENBLAS_LIBS)
else ifeq ($(BLAS),none)
    BLAS_DEFS   := -DMATRIX_NO_BLAS
else ifneq ($(BLAS),shim)
    $(error BLAS doit valoir auto, openblas, shim ou none (reçu : $(BLAS)))
endif
$(info [BLAS] backend = $(BLAS))

# nlohmann/json — header-only
JSON_INC := $(shell pkg-config --cflags nlohmann_json 2>/dev/null)
INC      += $(JSON_INC)

CXXFLAGS := $(STD) $(OPT) $(WARN) $(INC) $(BLAS_INC) $(BLAS_DEFS)
LDFLAGS  := $(LIBS) $(LSAN)

$(shell mkdir -p $(BIN))

# ══════════════════════════════════════════════════════════════════════════════
.PHONY: lib install uninstall all gpt2 llama smollm transformer benchmark tests blas-check FORCE \
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

# ── lowprec_tests  ───────────────────────────────────────────────────────────

tests: $(BIN)/lowprec_tests
$(BIN)/lowprec_tests: tests/lowprec_tests.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/lowprec_tests"
# 	$(BIN)/lowprec_tests

# ── backprop_test  ───────────────────────────────────────────────────────────

tests: $(BIN)/backprop_test
$(BIN)/backprop_test: tests/backprop_test.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/backprop_test"
# 	$(BIN)/backprop_test

# # ── profiler_probe_test  ───────────────────────────────────────────────────────────

tests: $(BIN)/profiler_probe_test
$(BIN)/profiler_probe_test: tests/profiler_probe_test.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/profiler_probe_test"
# 	$(BIN)/profiler_probe_test

# ── profiler_capability_test  ───────────────────────────────────────────────────────────

tests: $(BIN)/profiler_capability_test
$(BIN)/profiler_capability_test: tests/profiler_capability_test.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/profiler_capability_test"
# 	$(BIN)/profiler_capability_test

# ── profiler_workload_test  ───────────────────────────────────────────────────────────

tests: $(BIN)/profiler_workload_test
$(BIN)/profiler_workload_test: tests/profiler_workload_test.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/profiler_workload_test"
# 	$(BIN)/profiler_workload_test

# ── device_benchmark_test  ───────────────────────────────────────────────────────────

tests: $(BIN)/device_benchmark_test
$(BIN)/device_benchmark_test: tests/device_benchmark_test.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/device_benchmark_test"
# 	$(BIN)/device_benchmark_test

# ── hw_fingerprint_test  ───────────────────────────────────────────────────────────

tests: $(BIN)/hw_fingerprint_test
$(BIN)/hw_fingerprint_test: tests/hw_fingerprint_test.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/hw_fingerprint_test"
# 	$(BIN)/hw_fingerprint_test

# ── memory_profiler_test  ───────────────────────────────────────────────────────────

tests: $(BIN)/memory_profiler_test
$(BIN)/memory_profiler_test: tests/memory_profiler_test.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/memory_profiler_test"
# 	$(BIN)/memory_profiler_test

# ── algorithm_profiler_test  ───────────────────────────────────────────────────────────

tests: $(BIN)/algorithm_profiler_test
$(BIN)/algorithm_profiler_test: tests/algorithm_profiler_test.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/algorithm_profiler_test"
# 	$(BIN)/algorithm_profiler_test

# ── hyperparam_advisor_test  ───────────────────────────────────────────────────────────

tests: $(BIN)/hyperparam_advisor_test
$(BIN)/hyperparam_advisor_test: tests/hyperparam_advisor_test.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/hyperparam_advisor_test"
# 	$(BIN)/hyperparam_advisor_test

# ── profiler_memory_nohook_test  ───────────────────────────────────────────────────────────

tests: $(BIN)/profiler_memory_nohook_test
$(BIN)/profiler_memory_nohook_test: tests/profiler_memory_nohook_test.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/profiler_memory_nohook_test"
# 	$(BIN)/profiler_memory_nohook_test

# ── profiler_memory_test  ───────────────────────────────────────────────────────────

tests: $(BIN)/profiler_memory_test
$(BIN)/profiler_memory_test: tests/profiler_memory_test.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/profiler_memory_test"
# 	$(BIN)/profiler_memory_test

# ── profiler_advisor_test  ───────────────────────────────────────────────────────────

tests: $(BIN)/profiler_advisor_test
$(BIN)/profiler_advisor_test: tests/profiler_advisor_test.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/profiler_advisor_test"
# 	$(BIN)/profiler_advisor_test

# ── framework_bench_test  ───────────────────────────────────────────────────────────

tests: $(BIN)/framework_bench_test
$(BIN)/framework_bench_test: tests/framework_bench_test.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/framework_bench_test"
# 	$(BIN)/framework_bench_test

# ── profiler_suite_test  ───────────────────────────────────────────────────────────

tests: $(BIN)/profiler_suite_test
$(BIN)/profiler_suite_test: tests/profiler_suite_test.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/profiler_suite_test"
# 	$(BIN)/profiler_suite_test

# ── profiler_suite_nohook_test  ───────────────────────────────────────────────────────────

tests: $(BIN)/profiler_suite_nohook_test
$(BIN)/profiler_suite_nohook_test: tests/profiler_suite_nohook_test.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
	@echo "[RUN] $(BIN)/profiler_suite_nohook_test"
# 	$(BIN)/profiler_suite_nohook_test

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

# ── blas_check : prouve quel BLAS est réellement lié, vérifie les résultats, mesure les GFLOP/s ──
blas-check: $(BIN)/blas_check
	$(BIN)/blas_check $(if $(filter openblas shim none,$(BLAS)),--expect=$(BLAS))
$(BIN)/blas_check: tests/blas_check.cpp
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# ── Le backend BLAS fait partie des dépendances de chaque binaire ─────────────
# Sans ça, installer OpenBLAS (ou changer BLAS=) laisserait de vieux binaires « shim » en place.
BLAS_STAMP := $(BIN)/.blas_stamp
$(BLAS_STAMP): FORCE
	@echo "$(BLAS) $(BLAS_INC) $(OPENBLAS_LIBS) $(OPT)" | cmp -s - $@ || echo "$(BLAS) $(BLAS_INC) $(OPENBLAS_LIBS) $(OPT)" > $@
FORCE:
BIN_TARGETS := gpt2 smollm transformer server client basic_tests gcntest spmmtest coratrain contract diff_ops \
               matrix_tests scope_test support_tests lowprec_tests backprop_test profiler_probe_test \
               profiler_capability_test profiler_workload_test device_benchmark_test hw_fingerprint_test \
               memory_profiler_test algorithm_profiler_test hyperparam_advisor_test \
               profiler_memory_nohook_test profiler_memory_test profiler_advisor_test blas_check tensorf-gateway \
               framework_bench_test profiler_suite_test profiler_suite_nohook_test
# Headers are dependencies too: the profiler is header-only, so editing it must rebuild the tests.
HEADERS := $(shell find include -name '*.hpp' -o -name '*.h' 2>/dev/null)
$(addprefix $(BIN)/,$(BIN_TARGETS)): $(BLAS_STAMP) $(HEADERS)

# ══════════════════════════════════════════════════════════════════════════════
#  Bibliothèque statique + installation :  make && sudo make install
#  Ensuite, depuis n'importe quel dossier :
#      g++ -std=c++20 prog.cpp $(pkg-config --cflags --libs tensorf) -o prog
#  (ou à la main : -I$(PREFIX)/include/tensorf ... -L$(PREFIX)/lib -ltensorf)
#  PREFIX=$HOME/.local make install   → sans sudo (ajouter PKG_CONFIG_PATH=$HOME/.local/lib/pkgconfig)
# ══════════════════════════════════════════════════════════════════════════════
# TensorF est surtout des templates en en-têtes : ils sont compilés dans VOTRE programme, avec VOS options.
# libtensorf.a contient le code non-template (src/*.cpp, plus src/tensorf_lib.cpp : version + informations
# de build) ; le fichier tensorf.pc transporte le choix BLAS (-DTENSORF_USE_OPENBLAS, -lopenblas, -lpthread),
# qui DOIT être le même dans la bibliothèque et dans le programme. Statique ≠ plus rapide pour les gabarits ;
# la vitesse vient de BLAS, de -O2 et de -march.
PREFIX   ?= /usr/local
LIBDIR   ?= $(PREFIX)/lib
INCDIR   ?= $(PREFIX)/include/tensorf
VERSION  ?= 0.1.0
LIB_SRC  := $(wildcard src/*.cpp)
LIB_OBJ  := $(patsubst src/%.cpp,$(BIN)/obj/%.o,$(LIB_SRC))
LIB      := $(BIN)/libtensorf.a
INC_DIRS := . core nn data net profiler parallel tools
DESTDIR  ?=

lib: $(LIB)
$(BIN)/obj/%.o: src/%.cpp $(HEADERS) $(BLAS_STAMP)
	@mkdir -p $(dir $@)
	@echo "[CXX] $< → $@"
	$(CXX) $(CXXFLAGS) -fPIC -DTENSORF_VERSION=\"$(VERSION)\" -DTENSORF_BLAS_NAME=\"$(BLAS)\" -c $< -o $@
$(LIB): $(LIB_OBJ)
	@echo "[AR]  $@"
	ar rcs $@ $^
$(BIN)/tensorf.pc: $(BLAS_STAMP) Makefile FORCE   # rebuilt every time: PREFIX may have changed
	@echo "prefix=$(PREFIX)"                                                  >  $@
	@echo "libdir=$(LIBDIR)"                                                  >> $@
	@echo "includedir=$(INCDIR)"                                              >> $@
	@echo ""                                                                  >> $@
	@echo "Name: tensorf"                                                     >> $@
	@echo "Description: TensorF tensor / autograd / GNN framework (BLAS=$(BLAS))" >> $@
	@echo "Version: $(VERSION)"                                               >> $@
	@echo "Cflags: $(STD) $(OPT) $(BLAS_INC) $(BLAS_DEFS) $(foreach d,$(INC_DIRS),-I\$${includedir}/$(d))" >> $@
	@echo "Libs: -L\$${libdir} -ltensorf $(LIBS)"                              >> $@

install: $(LIB) $(BIN)/tensorf.pc
	install -d $(DESTDIR)$(LIBDIR) $(DESTDIR)$(LIBDIR)/pkgconfig $(DESTDIR)$(INCDIR)
	install -m 644 $(LIB) $(DESTDIR)$(LIBDIR)/libtensorf.a
	install -m 644 $(BIN)/tensorf.pc $(DESTDIR)$(LIBDIR)/pkgconfig/tensorf.pc
	cd include && find . \( -name '*.hpp' -o -name '*.h' \) -exec install -D -m 644 {} $(DESTDIR)$(INCDIR)/{} \;
	@echo "[INSTALL] $(DESTDIR)$(PREFIX)  (BLAS=$(BLAS))"
	@echo "  use:  g++ -std=c++20 prog.cpp \$$(pkg-config --cflags --libs tensorf) -o prog"
	@test "$(PREFIX)" != /usr/local -o -z "$(DESTDIR)" || true
uninstall:
	rm -rf $(DESTDIR)$(INCDIR) $(DESTDIR)$(LIBDIR)/libtensorf.a $(DESTDIR)$(LIBDIR)/pkgconfig/tensorf.pc

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
	@echo "── OpenBLAS (BLAS=$(BLAS))"
	@pkg-config --modversion openblas 2>/dev/null && echo "  OK" \
	 || echo "  MANQUANT — sudo apt install libopenblas-dev (sinon : shim lent)"
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
	@echo "  make blas-check     Quel BLAS est lié ? résultats + GFLOP/s"
	@echo "  make gateway        Web dashboard gateway (REST + WebSocket, gateway/src/main.cpp)"
	@echo "  make run-gateway    Build + lance le gateway (PORT=8080 par défaut)"
	@echo "  make clean          Supprime bin/"
	@echo "  make install-deps   Installe les libs système"
	@echo "  make check-deps     Vérifie les libs"
	@echo ""
	@echo "  DEBUG=1 make <cible>   ASan + UBSan + pas d'optimisation"
	@echo "  BLAS=auto|openblas|shim|none make <cible>   (défaut auto ; OPENBLAS_ROOT=... si hors système)"
	@echo "  SERVER_SRC=mon_serveur.cpp make server"
	@echo ""