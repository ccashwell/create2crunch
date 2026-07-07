/******** OpenCL wrapper for the bit-interleaved keccak core ********/

// OpenCL address-space qualifiers for the shared core (private locals need
// no qualifier in OpenCL C)
#define SPACE_CONSTANT __constant
#define SPACE_THREAD

//__CORE__//

__kernel void hashMessage(
  __constant uchar const *d_message,
  __constant uint const *d_nonce,
  __global volatile ulong *restrict solutions
) {
  uint gid = get_global_id(0);
  uint dn = d_nonce[0];

  if (hashAndCheck(gid, dn, d_message)) {
    solutions[0] = ((ulong)dn << 32) | (ulong)gid;
  }
}
