/******** Keccak-f[1600], bit-interleaved for 32-bit GPUs ********/

/**
* Bit-interleaved variant of keccak256.cl for GPUs without native 64-bit
* integer ALUs (notably Apple Silicon, where every ulong rotate/xor in the
* 64-bit kernel is emulated over multiple 32-bit operations).
*
* Each 64-bit keccak lane is stored as two 32-bit words: one packing the
* lane's even-numbered bits, one the odd-numbered bits. XOR/AND/NOT apply
* componentwise, and a 64-bit rotation by n becomes one 32-bit rotation of
* each component (with the components swapping places when n is odd) - all
* native single-cycle operations on a 32-bit ALU.
*
* Expects the host to define, in addition to the values used by the 64-bit
* kernel: A{i}_E/A{i}_O (pre-interleaved constant sponge lanes for
* i in {0,1,2,3,4,7,8,9,10,16}) and RC{i}_E/RC{i}_O for i in 0..23
* (pre-interleaved iota round constants).
*/

#define rotl32(x, s) rotate((uint)(x), (uint)(s))

// rotate a bit-interleaved lane (e, o) left by the constant n
#define ROLE(e, o, n) (((n) & 1) ? rotl32((o), (((n) + 1) >> 1)) : rotl32((e), ((n) >> 1)))
#define ROLO(e, o, n) (((n) & 1) ? rotl32((e), (((n) - 1) >> 1)) : rotl32((o), ((n) >> 1)))

// perfect unshuffle: gather the even bits of x into the low half-word and
// the odd bits into the high half-word
static inline uint unshuffle(uint x)
{
  uint t;
  t = (x ^ (x >> 1)) & 0x22222222u; x ^= t ^ (t << 1);
  t = (x ^ (x >> 2)) & 0x0c0c0c0cu; x ^= t ^ (t << 2);
  t = (x ^ (x >> 4)) & 0x00f000f0u; x ^= t ^ (t << 4);
  t = (x ^ (x >> 8)) & 0x0000ff00u; x ^= t ^ (t << 8);
  return x;
}

// inverse of unshuffle
static inline uint shuffle(uint x)
{
  uint t;
  t = (x ^ (x >> 8)) & 0x0000ff00u; x ^= t ^ (t << 8);
  t = (x ^ (x >> 4)) & 0x00f000f0u; x ^= t ^ (t << 4);
  t = (x ^ (x >> 2)) & 0x0c0c0c0cu; x ^= t ^ (t << 2);
  t = (x ^ (x >> 1)) & 0x22222222u; x ^= t ^ (t << 1);
  return x;
}

// interleave the 64-bit lane (lo, hi) into ae[i]/ao[i]
#define INTERLEAVE(i, lo, hi) \
lo = unshuffle(lo); \
hi = unshuffle(hi); \
ae[i] = (lo & 0xffffu) | (hi << 16); \
ao[i] = (lo >> 16) | (hi & 0xffff0000u);

#define theta_(m, n, x) \
te = be[m] ^ rotl32(bo[n], 1); \
to = bo[m] ^ be[n]; \
ae[x + 0] ^= te; \
ae[x + 5] ^= te; \
ae[x + 10] ^= te; \
ae[x + 15] ^= te; \
ae[x + 20] ^= te; \
ao[x + 0] ^= to; \
ao[x + 5] ^= to; \
ao[x + 10] ^= to; \
ao[x + 15] ^= to; \
ao[x + 20] ^= to;

#define theta() \
be[0] = ae[0] ^ ae[5] ^ ae[10] ^ ae[15] ^ ae[20]; \
bo[0] = ao[0] ^ ao[5] ^ ao[10] ^ ao[15] ^ ao[20]; \
be[1] = ae[1] ^ ae[6] ^ ae[11] ^ ae[16] ^ ae[21]; \
bo[1] = ao[1] ^ ao[6] ^ ao[11] ^ ao[16] ^ ao[21]; \
be[2] = ae[2] ^ ae[7] ^ ae[12] ^ ae[17] ^ ae[22]; \
bo[2] = ao[2] ^ ao[7] ^ ao[12] ^ ao[17] ^ ao[22]; \
be[3] = ae[3] ^ ae[8] ^ ae[13] ^ ae[18] ^ ae[23]; \
bo[3] = ao[3] ^ ao[8] ^ ao[13] ^ ao[18] ^ ao[23]; \
be[4] = ae[4] ^ ae[9] ^ ae[14] ^ ae[19] ^ ae[24]; \
bo[4] = ao[4] ^ ao[9] ^ ao[14] ^ ao[19] ^ ao[24]; \
theta_(4, 1, 0); \
theta_(0, 2, 1); \
theta_(1, 3, 2); \
theta_(2, 4, 3); \
theta_(3, 0, 4);

#define rhoPi_(m, n) \
te = be[0]; to = bo[0]; \
be[0] = ae[m]; bo[0] = ao[m]; \
ae[m] = ROLE(te, to, n); \
ao[m] = ROLO(te, to, n);

#define rhoPi() \
te = ae[1]; to = ao[1]; \
be[0] = ae[10]; bo[0] = ao[10]; \
ae[10] = rotl32(to, 1); ao[10] = te; \
rhoPi_(7, 3); \
rhoPi_(11, 6); \
rhoPi_(17, 10); \
rhoPi_(18, 15); \
rhoPi_(3, 21); \
rhoPi_(5, 28); \
rhoPi_(16, 36); \
rhoPi_(8, 45); \
rhoPi_(21, 55); \
rhoPi_(24, 2); \
rhoPi_(4, 14); \
rhoPi_(15, 27); \
rhoPi_(23, 41); \
rhoPi_(19, 56); \
rhoPi_(13, 8); \
rhoPi_(12, 25); \
rhoPi_(2, 43); \
rhoPi_(20, 62); \
rhoPi_(14, 18); \
rhoPi_(22, 39); \
rhoPi_(9, 61); \
rhoPi_(6, 20); \
rhoPi_(1, 44);

#define chi_(n) \
be[0] = ae[n + 0]; be[1] = ae[n + 1]; be[2] = ae[n + 2]; be[3] = ae[n + 3]; be[4] = ae[n + 4]; \
bo[0] = ao[n + 0]; bo[1] = ao[n + 1]; bo[2] = ao[n + 2]; bo[3] = ao[n + 3]; bo[4] = ao[n + 4]; \
ae[n + 0] = be[0] ^ ((~be[1]) & be[2]); \
ae[n + 1] = be[1] ^ ((~be[2]) & be[3]); \
ae[n + 2] = be[2] ^ ((~be[3]) & be[4]); \
ae[n + 3] = be[3] ^ ((~be[4]) & be[0]); \
ae[n + 4] = be[4] ^ ((~be[0]) & be[1]); \
ao[n + 0] = bo[0] ^ ((~bo[1]) & bo[2]); \
ao[n + 1] = bo[1] ^ ((~bo[2]) & bo[3]); \
ao[n + 2] = bo[2] ^ ((~bo[3]) & bo[4]); \
ao[n + 3] = bo[3] ^ ((~bo[4]) & bo[0]); \
ao[n + 4] = bo[4] ^ ((~bo[0]) & bo[1]);

#define chi() chi_(0); chi_(5); chi_(10); chi_(15); chi_(20);

#define iteration(re, ro) theta(); rhoPi(); chi(); ae[0] ^= re; ao[0] ^= ro;

#define hasTotal(d) ( \
  (!(d[0])) + (!(d[1])) + (!(d[2])) + (!(d[3])) + \
  (!(d[4])) + (!(d[5])) + (!(d[6])) + (!(d[7])) + \
  (!(d[8])) + (!(d[9])) + (!(d[10])) + (!(d[11])) + \
  (!(d[12])) + (!(d[13])) + (!(d[14])) + (!(d[15])) + \
  (!(d[16])) + (!(d[17])) + (!(d[18])) + (!(d[19])) \
>= TOTAL_ZEROES)

#if LEADING_ZEROES == 8
#define hasLeading(d) (!(((uint*)d)[0]) && !(((uint*)d)[1]))
#elif LEADING_ZEROES == 7
#define hasLeading(d) (!(((uint*)d)[0]) && !(((uint*)d)[1] & 0x00ffffffu))
#elif LEADING_ZEROES == 6
#define hasLeading(d) (!(((uint*)d)[0]) && !(((uint*)d)[1] & 0x0000ffffu))
#elif LEADING_ZEROES == 5
#define hasLeading(d) (!(((uint*)d)[0]) && !(((uint*)d)[1] & 0x000000ffu))
#elif LEADING_ZEROES == 4
#define hasLeading(d) (!(((uint*)d)[0]))
#elif LEADING_ZEROES == 3
#define hasLeading(d) (!(((uint*)d)[0] & 0x00ffffffu))
#elif LEADING_ZEROES == 2
#define hasLeading(d) (!(((uint*)d)[0] & 0x0000ffffu))
#elif LEADING_ZEROES == 1
#define hasLeading(d) (!(((uint*)d)[0] & 0x000000ffu))
#else
static inline bool hasLeading(uchar const *d)
{
#pragma unroll
  for (uint i = 0; i < LEADING_ZEROES; ++i) {
    if (d[i] != 0) return false;
  }
  return true;
}
#endif

__kernel void hashMessage(
  __constant uchar const *d_message,
  __constant uint const *d_nonce,
  __global volatile ulong *restrict solutions
) {
  uint ae[25];
  uint ao[25];
  uint be[5];
  uint bo[5];
  uint te, to;
  uint lo, hi;

  uint gid = get_global_id(0);
  uint dn = d_nonce[0];

  // constant sponge lanes, interleaved at kernel build time
  ae[0] = A0_E; ao[0] = A0_O;
  ae[1] = A1_E; ao[1] = A1_O;
  ae[2] = A2_E; ao[2] = A2_O;
  ae[3] = A3_E; ao[3] = A3_O;
  ae[4] = A4_E; ao[4] = A4_O;
  ae[7] = A7_E; ao[7] = A7_O;
  ae[8] = A8_E; ao[8] = A8_O;
  ae[9] = A9_E; ao[9] = A9_O;
  ae[10] = A10_E; ao[10] = A10_O;
  ae[16] = A16_E; ao[16] = A16_O;
  ae[11] = 0; ao[11] = 0;
  ae[12] = 0; ao[12] = 0;
  ae[13] = 0; ao[13] = 0;
  ae[14] = 0; ao[14] = 0;
  ae[15] = 0; ao[15] = 0;
  ae[17] = 0; ao[17] = 0;
  ae[18] = 0; ao[18] = 0;
  ae[19] = 0; ao[19] = 0;
  ae[20] = 0; ao[20] = 0;
  ae[21] = 0; ao[21] = 0;
  ae[22] = 0; ao[22] = 0;
  ae[23] = 0; ao[23] = 0;
  ae[24] = 0; ao[24] = 0;

  // lane 5 (sponge bytes 40..48): caller[19], salt_random_segment, nonce[0..3]
  lo = (uint)S_40
     | ((uint)d_message[0] << 8)
     | ((uint)d_message[1] << 16)
     | ((uint)d_message[2] << 24);
  hi = (uint)d_message[3]
     | ((gid & 0xffu) << 8)
     | (((gid >> 8) & 0xffu) << 16)
     | (((gid >> 16) & 0xffu) << 24);
  INTERLEAVE(5, lo, hi);

  // lane 6 (sponge bytes 48..56): nonce[3..8], init code hash bytes 0..3
  lo = (gid >> 24)
     | ((dn & 0xffu) << 8)
     | (((dn >> 8) & 0xffu) << 16)
     | (((dn >> 16) & 0xffu) << 24);
  hi = (dn >> 24)
     | ((uint)S_53 << 8)
     | ((uint)S_54 << 16)
     | ((uint)S_55 << 24);
  INTERLEAVE(6, lo, hi);

  iteration(RC0_E, RC0_O);   // iteration 1
  iteration(RC1_E, RC1_O);   // iteration 2
  iteration(RC2_E, RC2_O);   // iteration 3
  iteration(RC3_E, RC3_O);   // iteration 4
  iteration(RC4_E, RC4_O);   // iteration 5
  iteration(RC5_E, RC5_O);   // iteration 6
  iteration(RC6_E, RC6_O);   // iteration 7
  iteration(RC7_E, RC7_O);   // iteration 8
  iteration(RC8_E, RC8_O);   // iteration 9
  iteration(RC9_E, RC9_O);   // iteration 10
  iteration(RC10_E, RC10_O); // iteration 11
  iteration(RC11_E, RC11_O); // iteration 12
  iteration(RC12_E, RC12_O); // iteration 13
  iteration(RC13_E, RC13_O); // iteration 14
  iteration(RC14_E, RC14_O); // iteration 15
  iteration(RC15_E, RC15_O); // iteration 16
  iteration(RC16_E, RC16_O); // iteration 17
  iteration(RC17_E, RC17_O); // iteration 18
  iteration(RC18_E, RC18_O); // iteration 19
  iteration(RC19_E, RC19_O); // iteration 20
  iteration(RC20_E, RC20_O); // iteration 21
  iteration(RC21_E, RC21_O); // iteration 22
  iteration(RC22_E, RC22_O); // iteration 23

  // iteration 24 (partial): only the three lanes holding the address (state
  // bytes 12..32 = lanes 1 high, 2, 3) are computed; iota touches only lane
  // 0 and is skipped entirely

  // theta (full)
  be[0] = ae[0] ^ ae[5] ^ ae[10] ^ ae[15] ^ ae[20];
  bo[0] = ao[0] ^ ao[5] ^ ao[10] ^ ao[15] ^ ao[20];
  be[1] = ae[1] ^ ae[6] ^ ae[11] ^ ae[16] ^ ae[21];
  bo[1] = ao[1] ^ ao[6] ^ ao[11] ^ ao[16] ^ ao[21];
  be[2] = ae[2] ^ ae[7] ^ ae[12] ^ ae[17] ^ ae[22];
  bo[2] = ao[2] ^ ao[7] ^ ao[12] ^ ao[17] ^ ao[22];
  be[3] = ae[3] ^ ae[8] ^ ae[13] ^ ae[18] ^ ae[23];
  bo[3] = ao[3] ^ ao[8] ^ ao[13] ^ ao[18] ^ ao[23];
  be[4] = ae[4] ^ ae[9] ^ ae[14] ^ ae[19] ^ ae[24];
  bo[4] = ao[4] ^ ao[9] ^ ao[14] ^ ao[19] ^ ao[24];

  // theta applied to the pre-rho-pi sources of chi row 0
  uint a0e = ae[0] ^ be[4] ^ rotl32(bo[1], 1);
  uint a0o = ao[0] ^ bo[4] ^ be[1];
  uint a6e = ae[6] ^ be[0] ^ rotl32(bo[2], 1);
  uint a6o = ao[6] ^ bo[0] ^ be[2];
  uint a12e = ae[12] ^ be[1] ^ rotl32(bo[3], 1);
  uint a12o = ao[12] ^ bo[1] ^ be[3];
  uint a18e = ae[18] ^ be[2] ^ rotl32(bo[4], 1);
  uint a18o = ao[18] ^ bo[2] ^ be[4];
  uint a24e = ae[24] ^ be[3] ^ rotl32(bo[0], 1);
  uint a24o = ao[24] ^ bo[3] ^ be[0];

  // rho + pi images forming chi row 0 (a0 rotates by 0)
  uint b1e = ROLE(a6e, a6o, 44), b1o = ROLO(a6e, a6o, 44);
  uint b2e = ROLE(a12e, a12o, 43), b2o = ROLO(a12e, a12o, 43);
  uint b3e = ROLE(a18e, a18o, 21), b3o = ROLO(a18e, a18o, 21);
  uint b4e = ROLE(a24e, a24o, 14), b4o = ROLO(a24e, a24o, 14);

  // chi for lanes 1..3
  uint l1e = b1e ^ ((~b2e) & b3e), l1o = b1o ^ ((~b2o) & b3o);
  uint l2e = b2e ^ ((~b3e) & b4e), l2o = b2o ^ ((~b3o) & b4o);
  uint l3e = b3e ^ ((~b4e) & a0e), l3o = b3o ^ ((~b4o) & a0o);

  // de-interleave into the address bytes (state bytes 12..32)
  uint digestWords[5];
  digestWords[0] = shuffle((l1e >> 16) | (l1o & 0xffff0000u));
  digestWords[1] = shuffle((l2e & 0xffffu) | ((l2o & 0xffffu) << 16));
  digestWords[2] = shuffle((l2e >> 16) | (l2o & 0xffff0000u));
  digestWords[3] = shuffle((l3e & 0xffffu) | ((l3o & 0xffffu) << 16));
  digestWords[4] = shuffle((l3e >> 16) | (l3o & 0xffff0000u));

  uchar const *digest = (uchar const *)digestWords;

  // determine if the address meets the constraints
#ifdef PATTERN
  // pattern search: all provided constraints must hold simultaneously
  if (
    hasPattern(digest)
#if LEADING_ZEROES > 0
    && hasLeading(digest)
#endif
#if TOTAL_ZEROES <= 20
    && hasTotal(digest)
#endif
  ) {
    solutions[0] = ((ulong)dn << 32) | (ulong)gid;
  }
#else
  if (
    hasLeading(digest)
#if TOTAL_ZEROES <= 20
    || hasTotal(digest)
#endif
  ) {
    solutions[0] = ((ulong)dn << 32) | (ulong)gid;
  }
#endif
}
