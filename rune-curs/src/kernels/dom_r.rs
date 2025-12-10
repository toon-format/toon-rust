//! Embedded PTX for `DomR` kernel.

pub const DOMR_PTX: &str = r"
.version 6.0
.target sm_89
.address_size 64

.visible .entry domr_kernel(
    .param .u64 energy,
    .param .u64 coords,
    .param .u64 scores,
    .param .u32 n
) {
    .reg .pred p;
    .reg .f32 acc, dot, cx0,cx1,cx2,cx3,cx4,cx5,cx6,cx7, co;
    .reg .s32 idx, nval, o;
    .reg .u64 eptr, cptr, sptr, base;

    ld.param.u64 eptr, [energy];
    ld.param.u64 cptr, [coords];
    ld.param.u64 sptr, [scores];
    ld.param.u32 nval, [n];
    mov.u32 idx, %tid.x;
    mad.lo.s32 idx, %ctaid.x, %ntid.x, idx;
    setp.ge.s32 p, idx, nval;
    @p ret;

    // load coords[idx]
    mul.wide.s32 base, idx, 32;
    add.s64 base, cptr, base;
    ld.global.f32 cx0, [base+0];
    ld.global.f32 cx1, [base+4];
    ld.global.f32 cx2, [base+8];
    ld.global.f32 cx3, [base+12];
    ld.global.f32 cx4, [base+16];
    ld.global.f32 cx5, [base+20];
    ld.global.f32 cx6, [base+24];
    ld.global.f32 cx7, [base+28];

    mov.f32 acc, 0f00000000;
    mov.s32 o, 0;
L_loop:
    setp.ge.s32 p, o, nval;
    @p bra L_end;
    mul.wide.s32 base, o, 32;
    add.s64 base, cptr, base;
    ld.global.f32 dot, [base+0];
    mul.f32 dot, dot, cx0;
    ld.global.f32 co, [base+4];
    fma.rn.f32 dot, co, cx1, dot;
    ld.global.f32 co, [base+8];
    fma.rn.f32 dot, co, cx2, dot;
    ld.global.f32 co, [base+12];
    fma.rn.f32 dot, co, cx3, dot;
    ld.global.f32 co, [base+16];
    fma.rn.f32 dot, co, cx4, dot;
    ld.global.f32 co, [base+20];
    fma.rn.f32 dot, co, cx5, dot;
    ld.global.f32 co, [base+24];
    fma.rn.f32 dot, co, cx6, dot;
    ld.global.f32 co, [base+28];
    fma.rn.f32 dot, co, cx7, dot;

    ld.global.f32 co, [eptr + o*4];
    fma.rn.f32 acc, co, dot, acc;
    add.s32 o, o, 1;
    bra L_loop;
L_end:
    mul.wide.s32 base, idx, 4;
    add.s64 base, sptr, base;
    st.global.f32 [base], acc;
    ret;
}
";
