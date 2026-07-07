//! 2-way parallel keccak-f[1600] using the ARMv8.4 SHA3 NEON instructions.
//!
//! The SHA3 extension's EOR3 (three-way xor), RAX1 (xor-rotate-by-1), XAR
//! (xor-rotate) and BCAX (bit-clear-xor) instructions each operate on a full
//! 128-bit vector, i.e. two independent 64-bit keccak lanes at once. A single
//! hash stream cannot use that width - Apple's scalar pipeline already
//! saturates - so instead each vector register holds the same lane of two
//! *independent* sponge states, hashing two candidate salts per core
//! simultaneously.
//!
//! Callers must ensure the `sha3` target feature is present at runtime
//! (`std::arch::is_aarch64_feature_detected!("sha3")`); all Apple M-series
//! chips have it.

use crate::KECCAK_RC;
use core::arch::aarch64::*;

/// Destination index of the combined rho+pi step for each source lane:
/// lane (x, y) at flat index `x + 5y` moves to `(y, 2x + 3y)`.
const PI: [usize; 25] = [
    0, 10, 20, 5, 15, 16, 1, 11, 21, 6, 7, 17, 2, 12, 22, 23, 8, 18, 3, 13, 14, 24, 9, 19, 4,
];

/// Hash two 85-byte miner messages that differ only in their 6-byte nonces
/// (message bytes 47..53), returning each candidate's 20-byte address
/// (digest bytes 12..32).
///
/// `lanes` are the sponge rate lanes of the padded message with the nonce
/// bytes zeroed (see `sponge_lanes`); `salt_a`/`salt_b` supply the nonces.
///
/// # Safety
/// Requires the aarch64 `sha3` target feature at runtime.
#[target_feature(enable = "sha3")]
pub(crate) unsafe fn address_pair(
    lanes: &[u64; 17],
    salt_a: u64,
    salt_b: u64,
) -> ([u8; 20], [u8; 20]) {
    // splice each 6-byte nonce into lanes 5 and 6: nonce byte 0 is message
    // byte 47 (lane 5, bits 56..64), bytes 1..6 are message bytes 48..53
    // (lane 6, bits 0..40)
    let lane5 = |salt: u64| lanes[5] | ((salt & 0xff) << 56);
    let lane6 = |salt: u64| lanes[6] | ((salt >> 8) & 0xff_ffff_ffff);

    // state lane j holds candidate A in vector lane 0, candidate B in lane 1
    let mut a = [vdupq_n_u64(0); 25];
    for (j, &lane) in lanes.iter().enumerate() {
        a[j] = vdupq_n_u64(lane);
    }
    a[5] = vcombine_u64(vcreate_u64(lane5(salt_a)), vcreate_u64(lane5(salt_b)));
    a[6] = vcombine_u64(vcreate_u64(lane6(salt_a)), vcreate_u64(lane6(salt_b)));

    for &rc in &KECCAK_RC {
        // theta: column parities and D values
        let c0 = veor3q_u64(veor3q_u64(a[0], a[5], a[10]), a[15], a[20]);
        let c1 = veor3q_u64(veor3q_u64(a[1], a[6], a[11]), a[16], a[21]);
        let c2 = veor3q_u64(veor3q_u64(a[2], a[7], a[12]), a[17], a[22]);
        let c3 = veor3q_u64(veor3q_u64(a[3], a[8], a[13]), a[18], a[23]);
        let c4 = veor3q_u64(veor3q_u64(a[4], a[9], a[14]), a[19], a[24]);
        let d0 = vrax1q_u64(c4, c1);
        let d1 = vrax1q_u64(c0, c2);
        let d2 = vrax1q_u64(c1, c3);
        let d3 = vrax1q_u64(c2, c4);
        let d4 = vrax1q_u64(c3, c0);

        // theta + rho + pi: b[PI[i]] = rol(a[i] ^ d[i % 5], RHO[i]), where
        // XAR's immediate is the right-rotation 64 - RHO[i]
        let mut b = [vdupq_n_u64(0); 25];
        b[PI[0]] = veorq_u64(a[0], d0);
        b[PI[1]] = vxarq_u64::<63>(a[1], d1);
        b[PI[2]] = vxarq_u64::<2>(a[2], d2);
        b[PI[3]] = vxarq_u64::<36>(a[3], d3);
        b[PI[4]] = vxarq_u64::<37>(a[4], d4);
        b[PI[5]] = vxarq_u64::<28>(a[5], d0);
        b[PI[6]] = vxarq_u64::<20>(a[6], d1);
        b[PI[7]] = vxarq_u64::<58>(a[7], d2);
        b[PI[8]] = vxarq_u64::<9>(a[8], d3);
        b[PI[9]] = vxarq_u64::<44>(a[9], d4);
        b[PI[10]] = vxarq_u64::<61>(a[10], d0);
        b[PI[11]] = vxarq_u64::<54>(a[11], d1);
        b[PI[12]] = vxarq_u64::<21>(a[12], d2);
        b[PI[13]] = vxarq_u64::<39>(a[13], d3);
        b[PI[14]] = vxarq_u64::<25>(a[14], d4);
        b[PI[15]] = vxarq_u64::<23>(a[15], d0);
        b[PI[16]] = vxarq_u64::<19>(a[16], d1);
        b[PI[17]] = vxarq_u64::<49>(a[17], d2);
        b[PI[18]] = vxarq_u64::<43>(a[18], d3);
        b[PI[19]] = vxarq_u64::<56>(a[19], d4);
        b[PI[20]] = vxarq_u64::<46>(a[20], d0);
        b[PI[21]] = vxarq_u64::<62>(a[21], d1);
        b[PI[22]] = vxarq_u64::<3>(a[22], d2);
        b[PI[23]] = vxarq_u64::<8>(a[23], d3);
        b[PI[24]] = vxarq_u64::<50>(a[24], d4);

        // chi: a[x] = b[x] ^ (~b[x+1] & b[x+2]) per row
        for y in 0..5 {
            let r = 5 * y;
            a[r] = vbcaxq_u64(b[r], b[r + 2], b[r + 1]);
            a[r + 1] = vbcaxq_u64(b[r + 1], b[r + 3], b[r + 2]);
            a[r + 2] = vbcaxq_u64(b[r + 2], b[r + 4], b[r + 3]);
            a[r + 3] = vbcaxq_u64(b[r + 3], b[r], b[r + 4]);
            a[r + 4] = vbcaxq_u64(b[r + 4], b[r + 1], b[r]);
        }

        // iota
        a[0] = veorq_u64(a[0], vdupq_n_u64(rc));
    }

    // the address is digest bytes 12..32: the high half of lane 1, then
    // lanes 2 and 3 in full
    fn address(lane1: u64, lane2: u64, lane3: u64) -> [u8; 20] {
        let mut out = [0u8; 20];
        out[..4].copy_from_slice(&lane1.to_le_bytes()[4..]);
        out[4..12].copy_from_slice(&lane2.to_le_bytes());
        out[12..].copy_from_slice(&lane3.to_le_bytes());
        out
    }
    (
        address(
            vgetq_lane_u64::<0>(a[1]),
            vgetq_lane_u64::<0>(a[2]),
            vgetq_lane_u64::<0>(a[3]),
        ),
        address(
            vgetq_lane_u64::<1>(a[1]),
            vgetq_lane_u64::<1>(a[2]),
            vgetq_lane_u64::<1>(a[3]),
        ),
    )
}

#[cfg(test)]
mod tests {
    use crate::sponge_lanes;
    use keccak_asm::{Digest, Keccak256};

    /// Pin the vectorized 2-way keccak against the assembly-backed reference
    /// on the exact miner message layout.
    #[test]
    fn address_pair_matches_reference_digest() {
        if !std::arch::is_aarch64_feature_detected!("sha3") {
            eprintln!("skipping: no sha3 target feature at runtime");
            return;
        }

        // deterministic pseudorandom template (splitmix64), nonce bytes zeroed
        let mut template = [0u8; 85];
        let mut x = 0x243f6a8885a308d3u64;
        for byte in template.iter_mut() {
            x = x.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *byte = (z ^ (z >> 31)) as u8;
        }
        template[47..53].fill(0);
        let lanes = sponge_lanes(&template);

        let salt_pairs = [
            (0u64, 1u64),
            (2, 3),
            (0xdeadbeef, 0xcafef00d12),
            (0xffff_ffff_fffd, 0xffff_ffff_fffe),
            (0x0102_0304_0506, 0x60a0_b0c0_d0e0),
        ];
        for (salt_a, salt_b) in salt_pairs {
            let (address_a, address_b) = unsafe { super::address_pair(&lanes, salt_a, salt_b) };
            for (salt, address) in [(salt_a, address_a), (salt_b, address_b)] {
                let mut message = template;
                message[47..53].copy_from_slice(&salt.to_le_bytes()[..6]);
                let expected = Keccak256::digest(message);
                assert_eq!(
                    address,
                    expected[12..32],
                    "salt {salt:#x} produced a wrong address"
                );
            }
        }
    }
}
