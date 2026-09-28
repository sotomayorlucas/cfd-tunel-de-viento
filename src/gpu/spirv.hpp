// ============================================================================
//  gpu/spirv.hpp — emisor PROPIO de SPIR-V (sin glslang/shaderc/spirv-tools).
//
//  Dos capas:
//   * Module: escribe las palabras binarias del módulo por secciones en el orden
//     que exige la especificación (capacidades, extensiones, importaciones,
//     modelo de memoria, puntos de entrada, modos, depuración, decoraciones,
//     tipos/constantes/variables globales, funciones). Tipos y constantes se
//     deduplican (la especificación prohíbe tipos no agregados repetidos).
//   * Kernel: un DSL tipado mínimo para escribir compute shaders en C++ como si
//     fuera código escalar: valores F (float32), U (uint32), B (bool) con
//     operadores sobrecargados que emiten instrucciones en el bloque actual,
//     búferes de almacenamiento (f32/f16/u32/u8), memoria compartida, constantes
//     de empuje, variables locales, if/else estructurados (OpSelectionMerge),
//     operaciones de subgrupo y funciones GLSL.std.450. Los bucles se desenrollan
//     en tiempo de generación (el generador ES el preprocesador).
//
//  Todo shader se especializa AL GENERARLO (precisión, colisión, paridad, flags):
//  las constantes entran como literales y el compilador del controlador (NIR)
//  pliega lo demás. La validez se comprueba creando el pipeline en el
//  controlador y con tests de resultados en la GPU (tests/test_gpu_spirv.cpp).
// ============================================================================
#pragma once

#include "../core/config.hpp"

#include <initializer_list>
#include <map>
#include <string>
#include <vector>

namespace cfd::gpu::spv {

// ---- Constantes de la especificación SPIR-V 1.5 (sólo las usadas) ------------------------------
enum Op : u32 {
    OpName = 5, OpExtension = 10, OpExtInstImport = 11, OpExtInst = 12, OpMemoryModel = 14, OpEntryPoint = 15, OpExecutionMode = 16,
    OpCapability = 17, OpTypeVoid = 19, OpTypeBool = 20, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23, OpTypeArray = 28,
    OpTypeRuntimeArray = 29, OpTypeStruct = 30, OpTypePointer = 32, OpTypeFunction = 33, OpConstantTrue = 41, OpConstantFalse = 42,
    OpConstant = 43, OpConstantComposite = 44, OpFunction = 54, OpFunctionEnd = 56, OpVariable = 59, OpLoad = 61, OpStore = 62,
    OpAccessChain = 65, OpDecorate = 71, OpMemberDecorate = 72, OpCompositeConstruct = 80, OpCompositeExtract = 81,
    OpConvertFToU = 109, OpConvertFToS = 110, OpConvertSToF = 111, OpConvertUToF = 112, OpUConvert = 113, OpSConvert = 114,
    OpFConvert = 115, OpBitcast = 124, OpSNegate = 126, OpFNegate = 127, OpIAdd = 128, OpFAdd = 129, OpISub = 130, OpFSub = 131,
    OpIMul = 132, OpFMul = 133, OpUDiv = 134, OpSDiv = 135, OpFDiv = 136, OpUMod = 137, OpIsNan = 156, OpLogicalOr = 166,
    OpLogicalAnd = 167, OpLogicalNot = 168, OpSelect = 169, OpIEqual = 170, OpINotEqual = 171, OpUGreaterThan = 172,
    OpSGreaterThan = 173, OpUGreaterThanEqual = 174, OpSGreaterThanEqual = 175, OpULessThan = 176, OpSLessThan = 177,
    OpULessThanEqual = 178, OpSLessThanEqual = 179, OpFOrdEqual = 180, OpFOrdNotEqual = 182, OpFOrdLessThan = 184,
    OpFUnordLessThan = 185, OpFOrdGreaterThan = 186, OpFUnordGreaterThan = 187, OpFOrdLessThanEqual = 188,
    OpFUnordLessThanEqual = 189, OpFOrdGreaterThanEqual = 190, OpFUnordGreaterThanEqual = 191, OpShiftRightLogical = 194,
    OpShiftRightArithmetic = 195, OpShiftLeftLogical = 196, OpBitwiseOr = 197, OpBitwiseXor = 198, OpBitwiseAnd = 199, OpNot = 200,
    OpBitFieldUExtract = 203, OpBitCount = 205, OpControlBarrier = 224, OpMemoryBarrier = 225, OpAtomicIAdd = 234, OpAtomicOr = 241,
    OpSelectionMerge = 247, OpLabel = 248, OpBranch = 249, OpBranchConditional = 250, OpReturn = 253,
    OpGroupNonUniformElect = 333, OpGroupNonUniformAll = 334, OpGroupNonUniformAny = 335, OpGroupNonUniformBroadcastFirst = 338,
    OpGroupNonUniformBallot = 339, OpGroupNonUniformShuffle = 345, OpGroupNonUniformShuffleXor = 346, OpGroupNonUniformShuffleUp = 347,
    OpGroupNonUniformShuffleDown = 348, OpGroupNonUniformIAdd = 349, OpGroupNonUniformFAdd = 350, OpGroupNonUniformUMax = 357,
    OpGroupNonUniformBitwiseOr = 360,
};
enum Cap : u32 {
    CapShader = 1, CapFloat16 = 9, CapInt16 = 22, CapInt8 = 39, CapGroupNonUniform = 61, CapGroupNonUniformVote = 62,
    CapGroupNonUniformArithmetic = 63, CapGroupNonUniformBallot = 64, CapGroupNonUniformShuffle = 65, CapGroupNonUniformShuffleRelative = 66,
    CapStorageBuffer16BitAccess = 4433, CapStorageBuffer8BitAccess = 4448, CapDenormPreserve = 4464, CapRoundingModeRTE = 4467,
};
enum StorageClass : u32 { SC_Input = 1, SC_Uniform = 2, SC_Workgroup = 4, SC_Private = 6, SC_Function = 7, SC_PushConstant = 9, SC_StorageBuffer = 12 };
enum Decoration : u32 {
    DecBlock = 2, DecArrayStride = 6, DecBuiltIn = 11, DecRestrict = 19, DecAliased = 20, DecNonWritable = 24, DecNonReadable = 25,
    DecBinding = 33, DecDescriptorSet = 34, DecOffset = 35, DecFPRoundingMode = 39, DecNoContraction = 42,
};
enum BuiltIn : u32 {
    BI_NumWorkgroups = 24, BI_WorkgroupId = 26, BI_LocalInvocationId = 27, BI_GlobalInvocationId = 28, BI_LocalInvocationIndex = 29,
    BI_SubgroupSize = 36, BI_NumSubgroups = 38, BI_SubgroupId = 40, BI_SubgroupLocalInvocationId = 41,
};
enum ExecMode : u32 { EM_LocalSize = 17, EM_DenormPreserve = 4459, EM_RoundingModeRTE = 4462 };
enum Scope : u32 { ScopeDevice = 1, ScopeWorkgroup = 2, ScopeSubgroup = 3 };
enum Semantics : u32 { SemAcquireRelease = 0x8, SemUniformMemory = 0x40, SemWorkgroupMemory = 0x100 };
enum GLSL : u32 { GL_FAbs = 4, GL_Floor = 8, GL_Sqrt = 31, GL_InverseSqrt = 32, GL_FMin = 37, GL_UMin = 38, GL_FMax = 40, GL_UMax = 41,
                  GL_FClamp = 43, GL_Fma = 50, GL_PackHalf2x16 = 58, GL_UnpackHalf2x16 = 62 };

// ================================================================================================
//  Module: escritura binaria por secciones
// ================================================================================================
class Module {
public:
    Module();
    u32 id() { return next_id_++; }

    void capability(u32 c);
    void extension(const char* name);
    u32 glsl();                                     // importación de GLSL.std.450 (única)
    void entry_point(u32 fn, const char* name, const std::vector<u32>& iface);   // GLCompute
    void exec_mode(u32 fn, u32 mode, std::initializer_list<u32> args);
    void name(u32 target, const char* s);
    void decorate(u32 target, u32 dec, std::initializer_list<u32> args = {});
    void member_decorate(u32 st, u32 member, u32 dec, std::initializer_list<u32> args = {});

    // Tipos (deduplicados salvo las estructuras, que son siempre nuevas).
    u32 t_void();
    u32 t_bool();
    u32 t_int(u32 width, bool sign);
    u32 t_float(u32 width);
    u32 t_vec(u32 comp, u32 n);
    u32 t_array(u32 elem, u32 n);                   // longitud como constante u32
    u32 t_rtarray(u32 elem);
    u32 t_struct(const std::vector<u32>& members);
    u32 t_ptr(u32 sc, u32 type);
    u32 t_fn(u32 ret);

    // Constantes (deduplicadas).
    u32 c_bits(u32 type, u32 bits);
    u32 c_u32(u32 v) { return c_bits(t_int(32, false), v); }
    u32 c_i32(i32 v) { return c_bits(t_int(32, true), static_cast<u32>(v)); }
    u32 c_f32(float v);
    u32 c_bool(bool v);

    u32 global_var(u32 ptr_type, u32 sc);           // OpVariable en la sección global
    // Cuerpo de función (una sola función: main).
    void begin_main();
    u32 local_var(u32 ptr_type);                    // OpVariable Function (al prólogo)
    u32 op(u32 opcode, u32 type, std::initializer_list<u32> operands);
    u32 op(u32 opcode, u32 type, const std::vector<u32>& operands);
    void op0(u32 opcode, std::initializer_list<u32> operands);
    void label(u32 l);                              // abre un bloque
    void end_main();
    u32 main_fn() const { return main_; }

    std::vector<u32> assemble() const;              // cabecera + secciones

protected:
    static void emit(std::vector<u32>& s, u32 opcode, std::initializer_list<u32> w);
    static void emit(std::vector<u32>& s, u32 opcode, const std::vector<u32>& w);
    static void str(std::vector<u32>& w, const char* s);
    u32 cached(std::vector<u32>& section, u32 opcode, const std::vector<u32>& key_words, bool has_type);

    u32 next_id_ = 1;
    std::vector<u32> caps_, exts_, imports_, entries_, modes_, debug_, annot_, globals_, body_, prologue_;
    std::map<std::vector<u32>, u32> cache_;
    std::vector<u32> cap_list_;
    u32 glsl_ = 0, main_ = 0, first_label_pos_ = 0;
    bool in_main_ = false;
};

// ================================================================================================
//  Kernel: DSL tipado para compute shaders
// ================================================================================================
class Kernel;
Kernel& cur();   // kernel en construcción (uno por hilo)

struct B { u32 id = 0; };
struct U { u32 id = 0; U() = default; explicit U(u32 raw_id, int) : id(raw_id) {} U(u32 literal); };
struct F { u32 id = 0; F() = default; explicit F(u32 raw_id, int) : id(raw_id) {} F(float literal); F(double literal) : F(static_cast<float>(literal)) {} };

enum class Elem : u8 { F32, F16, U32, U16, U8 };

struct Buf {
    u32 var = 0;            // variable global (StorageBuffer)
    u32 elem_type = 0;      // tipo del elemento
    u32 elem_ptr = 0;       // puntero StorageBuffer al elemento
    Elem kind = Elem::F32;
};
struct Shared {
    u32 var = 0, elem_type = 0, elem_ptr = 0;
    bool is_float = true;
};
struct VarF { u32 var = 0; F get() const; void set(F v) const; };
struct VarU { u32 var = 0; U get() const; void set(U v) const; };
struct VarB { u32 var = 0; B get() const; void set(B v) const; };

struct KernelOptions {
    u32 local_size = 64;
    bool rte16 = true;              // modo de redondeo RTE para conversiones a f16 (si el dispositivo lo admite)
    bool denorm16 = true;           // preservar subnormales de f16 (paridad con F16C de la CPU)
    bool float16 = true;            // capacidad Float16 (tipos f16 en búferes)
    bool int8 = true;
};

class Kernel : public Module {
public:
    explicit Kernel(const KernelOptions& o);
    ~Kernel();
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;

    // ---- Recursos ----
    // Búfer de almacenamiento en (set 0, binding). readonly/writeonly → NonWritable/NonReadable.
    Buf buffer(u32 binding, Elem e, bool readonly = false, bool writeonly = false, bool restrict_ = true);
    Shared shared(u32 n, bool is_float);
    // Constantes de empuje: `n` palabras de 32 bits (u32 o f32 según la máscara de bits float_mask).
    void push_constants(u32 n_words, u32 float_mask);
    U pc_u(u32 i);
    F pc_f(u32 i);

    // ---- Entradas integradas ----
    U global_id(int axis = 0);
    U local_index();
    U workgroup_id(int axis = 0);
    U num_workgroups(int axis = 0);
    U subgroup_local_id();
    U subgroup_id();
    U subgroup_size();

    // ---- Memoria ----
    F ldf(const Buf& b, U idx);                 // f32 o f16 (→ f32)
    void stf(const Buf& b, U idx, F v);         // f32 o f16 (RTE)
    U ldu(const Buf& b, U idx);                 // u32/u16/u8 (extendido con ceros)
    void stu(const Buf& b, U idx, U v);
    F ldf(const Shared& s, U idx);
    void stf(const Shared& s, U idx, F v);
    U ldu(const Shared& s, U idx);
    void stu(const Shared& s, U idx, U v);
    void atomic_or(const Buf& b, U idx, U v);   // sobre u32
    void atomic_add(const Buf& b, U idx, U v);

    // ---- Variables locales ----
    VarF var(F init);
    VarU varu(U init);
    VarB varb(B init);

    // ---- Control de flujo estructurado ----
    template <class Fn> void if_(B c, Fn&& then) {
        const u32 lt = id(), lm = id();
        op0(OpSelectionMerge, {lm, 0});
        op0(OpBranchConditional, {c.id, lt, lm});
        label(lt);
        then();
        op0(OpBranch, {lm});
        label(lm);
    }
    template <class Fn, class Gn> void if_else(B c, Fn&& then, Gn&& els) {
        const u32 lt = id(), le = id(), lm = id();
        op0(OpSelectionMerge, {lm, 0});
        op0(OpBranchConditional, {c.id, lt, le});
        label(lt);
        then();
        op0(OpBranch, {lm});
        label(le);
        els();
        op0(OpBranch, {lm});
        label(lm);
    }

    // ---- Subgrupo / grupo de trabajo ----
    F sg_add(F v);
    U sg_add(U v);
    U sg_max(U v);
    U sg_or(U v);
    B sg_any(B v);
    B sg_all(B v);
    B sg_elect();
    F sg_shuffle_up(F v, u32 delta);
    U sg_shuffle_up(U v, u32 delta);
    F sg_shuffle_down(F v, u32 delta);
    F sg_shuffle_xor(F v, u32 mask);
    U sg_broadcast_first(U v);
    void barrier();                             // OpControlBarrier (grupo de trabajo, memoria compartida)

    // ---- Finalización ----
    std::vector<u32> finish();                  // cierra main y ensambla

    // Tipos frecuentes
    u32 tf, tu, ti, tb, tf16, tu8, tu16, tvoid, tuvec3;
    bool uses_f16 = false, uses_u8 = false, uses_u16 = false;

private:
    KernelOptions opt_;
    std::vector<u32> iface_;
    u32 pc_var_ = 0, pc_struct_ = 0, pc_float_mask_ = 0, pc_n_ = 0;
    u32 bi_[64] = {};
    u32 builtin_var(u32 bi, u32 type);
    U builtin_u(u32 bi);
    U builtin_u3(u32 bi, int axis);
    u32 elem_type(Elem e);
    u32 sg_scope_ = 0;
    Kernel* prev_ = nullptr;
};

// ---- Operadores (emiten en cur()) ------------------------------------------------------------
F operator+(F a, F b);
F operator-(F a, F b);
F operator*(F a, F b);
F operator/(F a, F b);
F operator-(F a);
inline F& operator+=(F& a, F b) { a = a + b; return a; }
inline F& operator-=(F& a, F b) { a = a - b; return a; }
inline F& operator*=(F& a, F b) { a = a * b; return a; }
U operator+(U a, U b);
U operator-(U a, U b);
U operator*(U a, U b);
U operator/(U a, U b);
U operator%(U a, U b);
U operator&(U a, U b);
U operator|(U a, U b);
U operator^(U a, U b);
U operator<<(U a, U b);
U operator>>(U a, U b);
inline U& operator+=(U& a, U b) { a = a + b; return a; }
inline U& operator|=(U& a, U b) { a = a | b; return a; }
B operator<(F a, F b);
B operator>(F a, F b);
B operator<=(F a, F b);
B operator>=(F a, F b);
B operator==(F a, F b);
B operator<(U a, U b);
B operator>(U a, U b);
B operator<=(U a, U b);
B operator>=(U a, U b);
B operator==(U a, U b);
B operator!=(U a, U b);
B operator&&(B a, B b);
B operator||(B a, B b);
B operator!(B a);
// Comparaciones no ordenadas (true si hay NaN): detección de divergencia como _CMP_NGT_UQ.
B ngt(F a, F b);    // !(a > b)
B nlt(F a, F b);    // !(a < b)
F select(B c, F a, F b);
U select(B c, U a, U b);
F fma(F a, F b, F c);       // GLSL Fma
F sqrt(F a);
F fmax(F a, F b);
F fmin(F a, F b);
F fabs(F a);
F clamp(F x, F lo, F hi);
U umin(U a, U b);
U umax(U a, U b);
F to_f(U a);                // u32 → f32
U to_u(F a);                // f32 → u32 (trunc)
F bits_f(U a);              // bitcast
U bits_u(F a);
B bit(U v, u32 b);          // (v >> b) & 1 != 0
B any_bits(U v, u32 mask);  // (v & mask) != 0
U ext(U v, u32 off, u32 count);   // OpBitFieldUExtract
// Dos f16 empaquetados en un u32 (GLSL PackHalf2x16/UnpackHalf2x16): lo = bits 0..15, hi = bits 16..31.
void unpack_half2(U w, F& lo, F& hi);
U pack_half2(F lo, F hi);

} // namespace cfd::gpu::spv
