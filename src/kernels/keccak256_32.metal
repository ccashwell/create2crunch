/******** Metal wrapper for the bit-interleaved keccak core ********/

#include <metal_stdlib>
using namespace metal;

// Metal address-space qualifiers for the shared core (stack locals and
// pointers to them live in the thread address space)
#define SPACE_CONSTANT constant
#define SPACE_THREAD thread

//__CORE__//

kernel void hashMessage(
  constant uchar *d_message [[buffer(0)]],
  constant uint *d_nonce [[buffer(1)]],
  device ulong *solutions [[buffer(2)]],
  uint gid [[thread_position_in_grid]]
) {
  uint dn = d_nonce[0];

  if (hashAndCheck(gid, dn, d_message)) {
    solutions[0] = ((ulong)dn << 32) | (ulong)gid;
  }
}
