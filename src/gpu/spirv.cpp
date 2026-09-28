// ============================================================================
//  gpu/spirv.cpp — emisor de SPIR-V (ver spirv.hpp).
//
//  Formato: cada instrucción = palabra 0 (nº de palabras << 16 | opcode) +
//  operandos. Cadenas literales: UTF-8 terminadas en 0, rellenas a 4 bytes.
// ============================================================================
#include "spirv.hpp"

#include <algorithm>
#include <bit>
#include <cstring>

namespace cfd::gpu::spv {

// ================================================================================================
//  Module
// ================================================================================================
Module::Module() = default;

void Module::emit(std::vector<u32>& s, u32 opcode, std::initializer_list<u32> w) {
    s.push_back((static_cast<u32>(w.size() + 1) << 16) | opcode);
    s.insert(s.end(), w.begin(), w.end());
}
void Module::emit(std::vector<u32>& s, u32 opcode, const std::vector<u32>& w) {
    s.push_back((static_cast<u32>(w.size() + 1) << 16) | opcode);
    s.insert(s.end(), w.begin(), w.end());
}
void Module::str(std::vector<u32>& w, const char* s) {
    // Bytes en orden little-endian dentro de cada palabra, con el 0 final y relleno de ceros hasta 4 bytes.
    const usize n = std::strlen(s) + 1;
    for (usize i = 0; i < n; i += 4) {
        u32 word = 0;
        for (usize j = 0; j < 4 && i + j < n; ++j) word |= static_cast<u32>(static_cast<u8>(s[i + j])) << (8 * j);
        w.push_back(word);
    }
}

void Module::capability(u32 c) {
    if (std::find(cap_list_.begin(), cap_list_.end(), c) != cap_list_.end()) return;
    cap_list_.push_back(c);
    emit(caps_, OpCapability, {c});
}
void Module::extension(const char* name) {
    std::vector<u32> w;
    str(w, name);
    emit(exts_, OpExtension, w);
}
u32 Module::glsl() {
    if (glsl_) return glsl_;
    glsl_ = id();
    std::vector<u32> w{glsl_};
    str(w, "GLSL.std.450");
    emit(imports_, OpExtInstImport, w);
    return glsl_;
}
void Module::entry_point(u32 fn, const char* nm, const std::vector<u32>& iface) {
    std::vector<u32> w{5u /*GLCompute*/, fn};
    str(w, nm);
    w.insert(w.end(), iface.begin(), iface.end());
    emit(entries_, OpEntryPoint, w);
}
void Module::exec_mode(u32 fn, u32 mode, std::initializer_list<u32> args) {
    std::vector<u32> w{fn, mode};
    w.insert(w.end(), args.begin(), args.end());
    emit(modes_, OpExecutionMode, w);
}
void Module::name(u32 target, const char* s) {
    std::vector<u32> w{target};
    str(w, s);
    emit(debug_, OpName, w);
}
void Module::decorate(u32 target, u32 dec, std::initializer_list<u32> args) {
    std::vector<u32> w{target, dec};
    w.insert(w.end(), args.begin(), args.end());
    emit(annot_, OpDecorate, w);
}
void Module::member_decorate(u32 st, u32 member, u32 dec, std::initializer_list<u32> args) {
    std::vector<u32> w{st, member, dec};
    w.insert(w.end(), args.begin(), args.end());
    emit(annot_, OpMemberDecorate, w);
}

// Tipos/constantes deduplicados: clave = (opcode, operandos sin el id de resultado).
u32 Module::cached(std::vector<u32>& section, u32 opcode, const std::vector<u32>& key_words, bool has_type) {
    std::vector<u32> key{opcode};
    key.insert(key.end(), key_words.begin(), key_words.end());
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;
    const u32 r = id();
    std::vector<u32> w;
    if (has_type) {   // [tipo, resultado, resto]
        w.push_back(key_words[0]);
        w.push_back(r);
        w.insert(w.end(), key_words.begin() + 1, key_words.end());
    } else {          // [resultado, resto]
        w.push_back(r);
        w.insert(w.end(), key_words.begin(), key_words.end());
    }
    emit(section, opcode, w);
    cache_.emplace(std::move(key), r);
    return r;
}

u32 Module::t_void() { return cached(globals_, OpTypeVoid, {}, false); }
u32 Module::t_bool() { return cached(globals_, OpTypeBool, {}, false); }
u32 Module::t_int(u32 width, bool sign) { return cached(globals_, OpTypeInt, {width, sign ? 1u : 0u}, false); }
u32 Module::t_float(u32 width) { return cached(globals_, OpTypeFloat, {width}, false); }
u32 Module::t_vec(u32 comp, u32 n) { return cached(globals_, OpTypeVector, {comp, n}, false); }
u32 Module::t_array(u32 elem, u32 n) { const u32 len = c_u32(n); return cached(globals_, OpTypeArray, {elem, len}, false); }
u32 Module::t_rtarray(u32 elem) { return cached(globals_, OpTypeRuntimeArray, {elem}, false); }
u32 Module::t_struct(const std::vector<u32>& members) {
    const u32 r = id();
    std::vector<u32> w{r};
    w.insert(w.end(), members.begin(), members.end());
    emit(globals_, OpTypeStruct, w);
    return r;
}
u32 Module::t_ptr(u32 sc, u32 type) { return cached(globals_, OpTypePointer, {sc, type}, false); }
u32 Module::t_fn(u32 ret) { return cached(globals_, OpTypeFunction, {ret}, false); }

u32 Module::c_bits(u32 type, u32 bits) { return cached(globals_, OpConstant, {type, bits}, true); }
u32 Module::c_f32(float v) { return c_bits(t_float(32), std::bit_cast<u32>(v)); }
u32 Module::c_bool(bool v) { return cached(globals_, v ? OpConstantTrue : OpConstantFalse, {t_bool()}, true); }

u32 Module::global_var(u32 ptr_type, u32 sc) {
    const u32 r = id();
    emit(globals_, OpVariable, {ptr_type, r, sc});
    return r;
}

void Module::begin_main() {
    const u32 tv = t_void();
    const u32 tfn = t_fn(tv);
    main_ = id();
    emit(body_, OpFunction, {tv, main_, 0u, tfn});
    const u32 l = id();
    emit(body_, OpLabel, {l});
    first_label_pos_ = static_cast<u32>(body_.size());
    in_main_ = true;
}
u32 Module::local_var(u32 ptr_type) {
    const u32 r = id();
    emit(prologue_, OpVariable, {ptr_type, r, u32(SC_Function)});
    return r;
}
u32 Module::op(u32 opcode, u32 type, std::initializer_list<u32> operands) {
    const u32 r = id();
    std::vector<u32> w{type, r};
    w.insert(w.end(), operands.begin(), operands.end());
    emit(body_, opcode, w);
    return r;
}
u32 Module::op(u32 opcode, u32 type, const std::vector<u32>& operands) {
    const u32 r = id();
    std::vector<u32> w{type, r};
    w.insert(w.end(), operands.begin(), operands.end());
    emit(body_, opcode, w);
    return r;
}
void Module::op0(u32 opcode, std::initializer_list<u32> operands) { emit(body_, opcode, operands); }
void Module::label(u32 l) { emit(body_, OpLabel, {l}); }
void Module::end_main() {
    if (!in_main_) return;
    emit(body_, OpReturn, {});
    emit(body_, OpFunctionEnd, {});
    in_main_ = false;
}

std::vector<u32> Module::assemble() const {
    std::vector<u32> m{0x07230203u, 0x00010500u /*SPIR-V 1.5*/, 0u /*generador*/, next_id_, 0u};
    auto add = [&](const std::vector<u32>& s) { m.insert(m.end(), s.begin(), s.end()); };
    add(caps_);
    add(exts_);
    add(imports_);
    m.push_back((3u << 16) | OpMemoryModel);
    m.push_back(0u);   // Logical
    m.push_back(1u);   // GLSL450
    add(entries_);
    add(modes_);
    add(debug_);
    add(annot_);
    add(globals_);
    // Función con las variables locales al principio del primer bloque (lo exige la especificación).
    m.insert(m.end(), body_.begin(), body_.begin() + first_label_pos_);
    add(prologue_);
    m.insert(m.end(), body_.begin() + first_label_pos_, body_.end());
    return m;
}

// ================================================================================================
//  Kernel
// ================================================================================================
namespace {
thread_local Kernel* g_cur = nullptr;
}
Kernel& cur() { return *g_cur; }

U::U(u32 literal) : id(cur().c_u32(literal)) {}
F::F(float literal) : id(cur().c_f32(literal)) {}

Kernel::Kernel(const KernelOptions& o) : opt_(o) {
    prev_ = g_cur;
    g_cur = this;
    capability(CapShader);
    tvoid = t_void();
    tb = t_bool();
    tu = t_int(32, false);
    ti = t_int(32, true);
    tf = t_float(32);
    tuvec3 = t_vec(tu, 3);
    tf16 = tu8 = tu16 = 0;
    sg_scope_ = c_u32(ScopeSubgroup);
    begin_main();
}

Kernel::~Kernel() { g_cur = prev_; }

u32 Kernel::elem_type(Elem e) {
    switch (e) {
        case Elem::F32: return tf;
        case Elem::U32: return tu;
        case Elem::F16:
            if (!tf16) {
                if (opt_.float16) capability(CapFloat16);
                capability(CapStorageBuffer16BitAccess);
                tf16 = t_float(16);
            }
            uses_f16 = true;
            return tf16;
        case Elem::U16:
            if (!tu16) { capability(CapInt16); capability(CapStorageBuffer16BitAccess); tu16 = t_int(16, false); }
            uses_u16 = true;
            return tu16;
        case Elem::U8:
            if (!tu8) {
                if (opt_.int8) capability(CapInt8);
                capability(CapStorageBuffer8BitAccess);
                tu8 = t_int(8, false);
            }
            uses_u8 = true;
            return tu8;
    }
    return tf;
}

Buf Kernel::buffer(u32 binding, Elem e, bool readonly, bool writeonly, bool restrict_) {
    Buf b;
    b.kind = e;
    b.elem_type = elem_type(e);
    const u32 stride = (e == Elem::F32 || e == Elem::U32) ? 4u : (e == Elem::U8 ? 1u : 2u);
    const std::vector<u32> key{OpTypeRuntimeArray, b.elem_type};
    const bool fresh = cache_.find(key) == cache_.end();
    const u32 rta = t_rtarray(b.elem_type);
    if (fresh) decorate(rta, DecArrayStride, {stride});
    const u32 st = t_struct({rta});
    decorate(st, DecBlock);
    member_decorate(st, 0, DecOffset, {0});
    if (readonly) member_decorate(st, 0, DecNonWritable);
    if (writeonly) member_decorate(st, 0, DecNonReadable);
    b.var = global_var(t_ptr(SC_StorageBuffer, st), SC_StorageBuffer);
    decorate(b.var, DecDescriptorSet, {0});
    decorate(b.var, DecBinding, {binding});
    if (restrict_) decorate(b.var, DecRestrict);
    b.elem_ptr = t_ptr(SC_StorageBuffer, b.elem_type);
    iface_.push_back(b.var);
    return b;
}

Shared Kernel::shared(u32 n, bool is_float) {
    Shared s;
    s.is_float = is_float;
    s.elem_type = is_float ? tf : tu;
    const u32 arr = t_array(s.elem_type, n);
    s.var = global_var(t_ptr(SC_Workgroup, arr), SC_Workgroup);
    s.elem_ptr = t_ptr(SC_Workgroup, s.elem_type);
    iface_.push_back(s.var);
    return s;
}

void Kernel::push_constants(u32 n_words, u32 float_mask) {
    std::vector<u32> mem(n_words);
    for (u32 i = 0; i < n_words; ++i) mem[i] = (float_mask >> i) & 1 ? tf : tu;
    pc_struct_ = t_struct(mem);
    decorate(pc_struct_, DecBlock);
    for (u32 i = 0; i < n_words; ++i) member_decorate(pc_struct_, i, DecOffset, {4 * i});
    pc_var_ = global_var(t_ptr(SC_PushConstant, pc_struct_), SC_PushConstant);
    pc_float_mask_ = float_mask;
    pc_n_ = n_words;
    iface_.push_back(pc_var_);
}
U Kernel::pc_u(u32 i) {
    const u32 p = op(OpAccessChain, t_ptr(SC_PushConstant, tu), {pc_var_, c_u32(i)});
    return U(op(OpLoad, tu, {p}), 0);
}
F Kernel::pc_f(u32 i) {
    const u32 p = op(OpAccessChain, t_ptr(SC_PushConstant, tf), {pc_var_, c_u32(i)});
    return F(op(OpLoad, tf, {p}), 0);
}

u32 Kernel::builtin_var(u32 bi, u32 type) {
    if (bi_[bi % 64]) return bi_[bi % 64];
    const u32 v = global_var(t_ptr(SC_Input, type), SC_Input);
    decorate(v, DecBuiltIn, {bi});
    iface_.push_back(v);
    bi_[bi % 64] = v;
    if (bi >= BI_SubgroupSize) capability(CapGroupNonUniform);
    return v;
}
U Kernel::builtin_u(u32 bi) { return U(op(OpLoad, tu, {builtin_var(bi, tu)}), 0); }
U Kernel::builtin_u3(u32 bi, int axis) {
    const u32 v = op(OpLoad, tuvec3, {builtin_var(bi, tuvec3)});
    return U(op(OpCompositeExtract, tu, {v, static_cast<u32>(axis)}), 0);
}
U Kernel::global_id(int axis) { return builtin_u3(BI_GlobalInvocationId, axis); }
U Kernel::local_index() { return builtin_u(BI_LocalInvocationIndex); }
U Kernel::workgroup_id(int axis) { return builtin_u3(BI_WorkgroupId, axis); }
U Kernel::num_workgroups(int axis) { return builtin_u3(BI_NumWorkgroups, axis); }
U Kernel::subgroup_local_id() { return builtin_u(BI_SubgroupLocalInvocationId); }
U Kernel::subgroup_id() { return builtin_u(BI_SubgroupId); }
U Kernel::subgroup_size() { return builtin_u(BI_SubgroupSize); }

F Kernel::ldf(const Buf& b, U idx) {
    const u32 p = op(OpAccessChain, b.elem_ptr, {b.var, c_u32(0), idx.id});
    const u32 v = op(OpLoad, b.elem_type, {p});
    if (b.kind == Elem::F16) return F(op(OpFConvert, tf, {v}), 0);
    if (b.kind == Elem::F32) return F(v, 0);
    return F(op(OpBitcast, tf, {v}), 0);
}
void Kernel::stf(const Buf& b, U idx, F v) {
    const u32 p = op(OpAccessChain, b.elem_ptr, {b.var, c_u32(0), idx.id});
    u32 x = v.id;
    if (b.kind == Elem::F16) x = op(OpFConvert, tf16, {v.id});
    else if (b.kind == Elem::U32) x = op(OpBitcast, tu, {v.id});
    op0(OpStore, {p, x});
}
U Kernel::ldu(const Buf& b, U idx) {
    const u32 p = op(OpAccessChain, b.elem_ptr, {b.var, c_u32(0), idx.id});
    const u32 v = op(OpLoad, b.elem_type, {p});
    if (b.kind == Elem::U8 || b.kind == Elem::U16) return U(op(OpUConvert, tu, {v}), 0);
    if (b.kind == Elem::F32) return U(op(OpBitcast, tu, {v}), 0);
    return U(v, 0);
}
void Kernel::stu(const Buf& b, U idx, U v) {
    const u32 p = op(OpAccessChain, b.elem_ptr, {b.var, c_u32(0), idx.id});
    u32 x = v.id;
    if (b.kind == Elem::U8 || b.kind == Elem::U16) x = op(OpUConvert, b.elem_type, {v.id});
    else if (b.kind == Elem::F32) x = op(OpBitcast, tf, {v.id});
    op0(OpStore, {p, x});
}
F Kernel::ldf(const Shared& s, U idx) {
    const u32 p = op(OpAccessChain, s.elem_ptr, {s.var, idx.id});
    const u32 v = op(OpLoad, s.elem_type, {p});
    return s.is_float ? F(v, 0) : F(op(OpBitcast, tf, {v}), 0);
}
void Kernel::stf(const Shared& s, U idx, F v) {
    const u32 p = op(OpAccessChain, s.elem_ptr, {s.var, idx.id});
    op0(OpStore, {p, s.is_float ? v.id : op(OpBitcast, tu, {v.id})});
}
U Kernel::ldu(const Shared& s, U idx) {
    const u32 p = op(OpAccessChain, s.elem_ptr, {s.var, idx.id});
    const u32 v = op(OpLoad, s.elem_type, {p});
    return s.is_float ? U(op(OpBitcast, tu, {v}), 0) : U(v, 0);
}
void Kernel::stu(const Shared& s, U idx, U v) {
    const u32 p = op(OpAccessChain, s.elem_ptr, {s.var, idx.id});
    op0(OpStore, {p, s.is_float ? op(OpBitcast, tf, {v.id}) : v.id});
}
void Kernel::atomic_or(const Buf& b, U idx, U v) {
    const u32 p = op(OpAccessChain, b.elem_ptr, {b.var, c_u32(0), idx.id});
    op(OpAtomicOr, tu, {p, c_u32(ScopeDevice), c_u32(0), v.id});
}
void Kernel::atomic_add(const Buf& b, U idx, U v) {
    const u32 p = op(OpAccessChain, b.elem_ptr, {b.var, c_u32(0), idx.id});
    op(OpAtomicIAdd, tu, {p, c_u32(ScopeDevice), c_u32(0), v.id});
}

VarF Kernel::var(F init) { VarF v{local_var(t_ptr(SC_Function, tf))}; v.set(init); return v; }
VarU Kernel::varu(U init) { VarU v{local_var(t_ptr(SC_Function, tu))}; v.set(init); return v; }
VarB Kernel::varb(B init) { VarB v{local_var(t_ptr(SC_Function, tb))}; v.set(init); return v; }
F VarF::get() const { return F(cur().op(OpLoad, cur().tf, {var}), 0); }
void VarF::set(F v) const { cur().op0(OpStore, {var, v.id}); }
U VarU::get() const { return U(cur().op(OpLoad, cur().tu, {var}), 0); }
void VarU::set(U v) const { cur().op0(OpStore, {var, v.id}); }
B VarB::get() const { return B{cur().op(OpLoad, cur().tb, {var})}; }
void VarB::set(B v) const { cur().op0(OpStore, {var, v.id}); }

F Kernel::sg_add(F v) { capability(CapGroupNonUniformArithmetic); return F(op(OpGroupNonUniformFAdd, tf, {sg_scope_, 0u, v.id}), 0); }
U Kernel::sg_add(U v) { capability(CapGroupNonUniformArithmetic); return U(op(OpGroupNonUniformIAdd, tu, {sg_scope_, 0u, v.id}), 0); }
U Kernel::sg_max(U v) { capability(CapGroupNonUniformArithmetic); return U(op(OpGroupNonUniformUMax, tu, {sg_scope_, 0u, v.id}), 0); }
U Kernel::sg_or(U v) { capability(CapGroupNonUniformArithmetic); return U(op(OpGroupNonUniformBitwiseOr, tu, {sg_scope_, 0u, v.id}), 0); }
B Kernel::sg_any(B v) { capability(CapGroupNonUniformVote); return B{op(OpGroupNonUniformAny, tb, {sg_scope_, v.id})}; }
B Kernel::sg_all(B v) { capability(CapGroupNonUniformVote); return B{op(OpGroupNonUniformAll, tb, {sg_scope_, v.id})}; }
B Kernel::sg_elect() { capability(CapGroupNonUniform); return B{op(OpGroupNonUniformElect, tb, {sg_scope_})}; }
F Kernel::sg_shuffle_up(F v, u32 d) {
    capability(CapGroupNonUniformShuffleRelative);
    return F(op(OpGroupNonUniformShuffleUp, tf, {sg_scope_, v.id, c_u32(d)}), 0);
}
U Kernel::sg_shuffle_up(U v, u32 d) {
    capability(CapGroupNonUniformShuffleRelative);
    return U(op(OpGroupNonUniformShuffleUp, tu, {sg_scope_, v.id, c_u32(d)}), 0);
}
F Kernel::sg_shuffle_down(F v, u32 d) {
    capability(CapGroupNonUniformShuffleRelative);
    return F(op(OpGroupNonUniformShuffleDown, tf, {sg_scope_, v.id, c_u32(d)}), 0);
}
F Kernel::sg_shuffle_xor(F v, u32 m) {
    capability(CapGroupNonUniformShuffle);
    return F(op(OpGroupNonUniformShuffleXor, tf, {sg_scope_, v.id, c_u32(m)}), 0);
}
U Kernel::sg_broadcast_first(U v) { capability(CapGroupNonUniformBallot); return U(op(OpGroupNonUniformBroadcastFirst, tu, {sg_scope_, v.id}), 0); }
void Kernel::barrier() {
    op0(OpControlBarrier, {c_u32(ScopeWorkgroup), c_u32(ScopeWorkgroup), c_u32(SemAcquireRelease | SemWorkgroupMemory)});
}

std::vector<u32> Kernel::finish() {
    end_main();
    entry_point(main_, "main", iface_);
    exec_mode(main_, EM_LocalSize, {opt_.local_size, 1, 1});
    if (uses_f16) {
        if (opt_.rte16) { capability(CapRoundingModeRTE); exec_mode(main_, EM_RoundingModeRTE, {16}); }
        if (opt_.denorm16) { capability(CapDenormPreserve); exec_mode(main_, EM_DenormPreserve, {16}); }
    }
    return assemble();
}

// ================================================================================================
//  Operadores
// ================================================================================================
namespace {
F fbin(u32 opc, F a, F b) { return F(cur().op(opc, cur().tf, {a.id, b.id}), 0); }
U ubin(u32 opc, U a, U b) { return U(cur().op(opc, cur().tu, {a.id, b.id}), 0); }
B fcmp(u32 opc, F a, F b) { return B{cur().op(opc, cur().tb, {a.id, b.id})}; }
B ucmp(u32 opc, U a, U b) { return B{cur().op(opc, cur().tb, {a.id, b.id})}; }
F gl1(u32 inst, F a) { Kernel& k = cur(); return F(k.op(OpExtInst, k.tf, {k.glsl(), inst, a.id}), 0); }
F gl2(u32 inst, F a, F b) { Kernel& k = cur(); return F(k.op(OpExtInst, k.tf, {k.glsl(), inst, a.id, b.id}), 0); }
F gl3(u32 inst, F a, F b, F c) { Kernel& k = cur(); return F(k.op(OpExtInst, k.tf, {k.glsl(), inst, a.id, b.id, c.id}), 0); }
} // namespace

F operator+(F a, F b) { return fbin(OpFAdd, a, b); }
F operator-(F a, F b) { return fbin(OpFSub, a, b); }
F operator*(F a, F b) { return fbin(OpFMul, a, b); }
F operator/(F a, F b) { return fbin(OpFDiv, a, b); }
F operator-(F a) { return F(cur().op(OpFNegate, cur().tf, {a.id}), 0); }
U operator+(U a, U b) { return ubin(OpIAdd, a, b); }
U operator-(U a, U b) { return ubin(OpISub, a, b); }
U operator*(U a, U b) { return ubin(OpIMul, a, b); }
U operator/(U a, U b) { return ubin(OpUDiv, a, b); }
U operator%(U a, U b) { return ubin(OpUMod, a, b); }
U operator&(U a, U b) { return ubin(OpBitwiseAnd, a, b); }
U operator|(U a, U b) { return ubin(OpBitwiseOr, a, b); }
U operator^(U a, U b) { return ubin(OpBitwiseXor, a, b); }
U operator<<(U a, U b) { return ubin(OpShiftLeftLogical, a, b); }
U operator>>(U a, U b) { return ubin(OpShiftRightLogical, a, b); }
B operator<(F a, F b) { return fcmp(OpFOrdLessThan, a, b); }
B operator>(F a, F b) { return fcmp(OpFOrdGreaterThan, a, b); }
B operator<=(F a, F b) { return fcmp(OpFOrdLessThanEqual, a, b); }
B operator>=(F a, F b) { return fcmp(OpFOrdGreaterThanEqual, a, b); }
B operator==(F a, F b) { return fcmp(OpFOrdEqual, a, b); }
B operator<(U a, U b) { return ucmp(OpULessThan, a, b); }
B operator>(U a, U b) { return ucmp(OpUGreaterThan, a, b); }
B operator<=(U a, U b) { return ucmp(OpULessThanEqual, a, b); }
B operator>=(U a, U b) { return ucmp(OpUGreaterThanEqual, a, b); }
B operator==(U a, U b) { return ucmp(OpIEqual, a, b); }
B operator!=(U a, U b) { return ucmp(OpINotEqual, a, b); }
B operator&&(B a, B b) { return B{cur().op(OpLogicalAnd, cur().tb, {a.id, b.id})}; }
B operator||(B a, B b) { return B{cur().op(OpLogicalOr, cur().tb, {a.id, b.id})}; }
B operator!(B a) { return B{cur().op(OpLogicalNot, cur().tb, {a.id})}; }
B ngt(F a, F b) { return fcmp(OpFUnordLessThanEqual, a, b); }
B nlt(F a, F b) { return fcmp(OpFUnordGreaterThanEqual, a, b); }
F select(B c, F a, F b) { return F(cur().op(OpSelect, cur().tf, {c.id, a.id, b.id}), 0); }
U select(B c, U a, U b) { return U(cur().op(OpSelect, cur().tu, {c.id, a.id, b.id}), 0); }
F fma(F a, F b, F c) { return gl3(GL_Fma, a, b, c); }
F sqrt(F a) { return gl1(GL_Sqrt, a); }
F fmax(F a, F b) { return gl2(GL_FMax, a, b); }
F fmin(F a, F b) { return gl2(GL_FMin, a, b); }
F fabs(F a) { return gl1(GL_FAbs, a); }
F clamp(F x, F lo, F hi) { return gl3(GL_FClamp, x, lo, hi); }
U umin(U a, U b) { Kernel& k = cur(); return U(k.op(OpExtInst, k.tu, {k.glsl(), u32(GL_UMin), a.id, b.id}), 0); }
U umax(U a, U b) { Kernel& k = cur(); return U(k.op(OpExtInst, k.tu, {k.glsl(), u32(GL_UMax), a.id, b.id}), 0); }
F to_f(U a) { return F(cur().op(OpConvertUToF, cur().tf, {a.id}), 0); }
U to_u(F a) { return U(cur().op(OpConvertFToU, cur().tu, {a.id}), 0); }
F bits_f(U a) { return F(cur().op(OpBitcast, cur().tf, {a.id}), 0); }
U bits_u(F a) { return U(cur().op(OpBitcast, cur().tu, {a.id}), 0); }
B bit(U v, u32 b) { return (v & U(1u << b)) != U(0u); }
B any_bits(U v, u32 mask) { return (v & U(mask)) != U(0u); }
void unpack_half2(U w, F& lo, F& hi) {
    Kernel& k = cur();
    const u32 v2 = k.t_vec(k.tf, 2);
    const u32 r = k.op(OpExtInst, v2, {k.glsl(), u32(GL_UnpackHalf2x16), w.id});
    lo = F(k.op(OpCompositeExtract, k.tf, {r, 0u}), 0);
    hi = F(k.op(OpCompositeExtract, k.tf, {r, 1u}), 0);
}
U pack_half2(F lo, F hi) {
    Kernel& k = cur();
    const u32 v2 = k.t_vec(k.tf, 2);
    const u32 c = k.op(OpCompositeConstruct, v2, {lo.id, hi.id});
    return U(k.op(OpExtInst, k.tu, {k.glsl(), u32(GL_PackHalf2x16), c}), 0);
}
U ext(U v, u32 off, u32 count) { Kernel& k = cur(); return U(k.op(OpBitFieldUExtract, k.tu, {v.id, k.c_u32(off), k.c_u32(count)}), 0); }

} // namespace cfd::gpu::spv
