# =============================================================================
#  Túnel de viento CFD — C++23 puro, afinado para Intel Core Ultra 7 155H
#  (Meteor Lake: Redwood Cove P-cores + Crestmont E-cores; AVX2/FMA/F16C/BMI2,
#   sin AVX-512).  Única dependencia del sistema: Xlib + XShm para la ventana.
#
#    make            → build optimizado (LTO)            → build/cfd
#    make pgo        → optimización guiada por perfil (2 pasadas + benchmark)
#    make unity      → "jumbo build": todo en una unidad de traducción
#    make debug      → -O1 -g + ASan/UBSan
#    make test       → compila y ejecuta todos los tests (tests/test_*.cpp)
#    make bench      → benchmark del solver (MLUPS por nº de hilos y precisión)
# =============================================================================
CXX      ?= g++
BUILD    ?= build
TARGET   := $(BUILD)/cfd

# -march=native en esta máquina = meteorlake (equivale a alderlake + extras). Se deja
# native para que el mismo árbol compile óptimo en otra CPU; MARCH=meteorlake para fijarlo.
MARCH    ?= native
WARN     := -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers
OPT      := -O3 -march=$(MARCH) -mtune=$(MARCH) \
            -fno-plt -fno-semantic-interposition -fomit-frame-pointer \
            -funroll-loops -fno-math-errno -fno-trapping-math -fno-signed-zeros \
            -ffp-contract=fast -fno-rtti -fstrict-aliasing \
            -fipa-pta -fdevirtualize-at-ltrans
LTO      := -flto=auto -fuse-linker-plugin
CXXFLAGS ?= -std=c++23 $(OPT) $(WARN) -DNDEBUG -pthread -MMD -MP -Isrc
LDFLAGS  ?= -pthread $(LTO) -Wl,-O1,--as-needed,--gc-sections,-z,now
LDLIBS   := -lX11 -lXext -ldl   # -ldl: libvulkan.so.1 se carga con dlopen (backend iGPU, src/gpu)

APP_SRC  := $(wildcard src/core/*.cpp src/geom/*.cpp src/models/*.cpp src/lbm/*.cpp src/gpu/*.cpp \
                       src/render/*.cpp src/ui/*.cpp src/platform/*.cpp src/app/*.cpp)
LIB_SRC  := $(filter-out src/app/main.cpp,$(APP_SRC))
APP_OBJ  := $(APP_SRC:%.cpp=$(BUILD)/obj/%.o)
LIB_OBJ  := $(LIB_SRC:%.cpp=$(BUILD)/obj/%.o)
TEST_SRC := $(wildcard tests/test_*.cpp)
TEST_BIN := $(TEST_SRC:tests/%.cpp=$(BUILD)/tests/%)
TOOL_SRC := $(wildcard tools/*.cpp)
TOOL_BIN := $(TOOL_SRC:tools/%.cpp=$(BUILD)/tools/%)

.PHONY: all clean test bench pgo unity debug tools
all: $(TARGET)

$(BUILD)/obj/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(LTO) $(PGOFLAGS) -c $< -o $@

$(TARGET): $(APP_OBJ)
	$(CXX) $(CXXFLAGS) $(PGOFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD)/tests/%: tests/%.cpp $(LIB_OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(LTO) $< $(LIB_OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

$(BUILD)/tools/%: tools/%.cpp $(LIB_OBJ)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(LTO) $< $(LIB_OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

tools: $(TOOL_BIN)

test: $(TEST_BIN)
	@set -e; for t in $(TEST_BIN); do echo "== $$t"; $$t; done

bench: $(BUILD)/tools/bench_lbm
	$(BUILD)/tools/bench_lbm

# Jumbo/unity build: una sola TU → el compilador ve todo el programa sin depender de LTO.
# x11.cpp va al final: Xlib define macros (None, Bool, Status...) que romperían el resto.
# -Wno-subobject-linkage: en una sola TU los tipos de espacios anónimos usados como miembros
# de clases con enlace externo (Solver::Impl) avisan aunque sea inocuo.
UNITY_SRC := $(filter-out src/platform/x11.cpp,$(APP_SRC)) src/platform/x11.cpp
unity:
	@mkdir -p $(BUILD)
	@printf '%s\n' $(foreach f,$(UNITY_SRC),'#include "../$(f)"') > $(BUILD)/unity.cpp
	$(CXX) $(CXXFLAGS) -Wno-subobject-linkage $(BUILD)/unity.cpp -o $(BUILD)/cfd_unity $(LDFLAGS) $(LDLIBS)

# PGO: instrumentar → ejecutar carga representativa (benchmark headless) → recompilar.
# -fprofile-update=single: contadores no atómicos (con =atomic los 20 hilos se pelean por las mismas
# líneas de caché en los bucles calientes y la carga de perfilado se hace muchísimo más lenta);
# -fprofile-correction: suaviza los recuentos inconsistentes que dejan las carreras entre hilos
# (sin ella GCC rechaza el perfil con "corrupted profile info"). -Wno-coverage-mismatch: si una fuente
# cambia entre las dos pasadas, esa función se compila sin perfil en vez de abortar.
pgo:
	$(MAKE) clean-obj
	$(MAKE) BUILD=$(BUILD) PGOFLAGS="-fprofile-generate -fprofile-update=single -fprofile-dir=$(abspath $(BUILD))/pgo" $(TARGET)
	$(TARGET) --bench-pgo || true
	$(MAKE) clean-obj
	$(MAKE) BUILD=$(BUILD) PGOFLAGS="-fprofile-use -fprofile-correction -fprofile-partial-training -fprofile-dir=$(abspath $(BUILD))/pgo -Wno-missing-profile -Wno-coverage-mismatch" $(TARGET)

debug:
	$(MAKE) BUILD=build-debug OPT="-O1 -g -march=$(MARCH) -fno-omit-frame-pointer -fsanitize=address,undefined" \
	        LTO="" CXXFLAGS="-std=c++23 -O1 -g -march=$(MARCH) -fsanitize=address,undefined -fno-omit-frame-pointer $(WARN) -pthread -MMD -MP -Isrc" \
	        LDFLAGS="-pthread -fsanitize=address,undefined"

clean-obj:
	rm -rf $(BUILD)/obj

clean:
	rm -rf $(BUILD) build-debug

-include $(APP_OBJ:.o=.d)
