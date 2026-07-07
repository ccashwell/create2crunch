# create2crunch

> A Rust program for finding salts that create gas-efficient Ethereum addresses via CREATE2.

Provide three arguments: a factory address (or contract that will call CREATE2), a caller address (for factory addresses that require it as a protection against frontrunning), and the keccak-256 hash of the initialization code of the contract that the factory will deploy. 
(The example below references the `Create2Factory`'s address on one of the 21 chains where it has been deployed to.)

Live `Create2Factory` contracts can be found [here](https://blockscan.com/address/0x0000000000ffe8b47b3e2130213b802212439497).

```sh
$ git clone https://github.com/0age/create2crunch
$ cd create2crunch
$ export FACTORY="0x0000000000ffe8b47b3e2130213b802212439497"
$ export CALLER="<YOUR_DEPLOYER_ADDRESS_OF_CHOICE_GOES_HERE>"
$ export INIT_CODE_HASH="<HASH_OF_YOUR_CONTRACT_INIT_CODE_GOES_HERE>"
$ cargo run --release $FACTORY $CALLER $INIT_CODE_HASH
```

For each efficient address found, the salt, resultant addresses, and value *(i.e. approximate rarity)* will be written to `efficient_addresses.txt`. Verify that one of the salts actually results in the intended address before getting in too deep - ideally, the CREATE2 factory will have a view method for checking what address you'll get for submitting a particular salt. Be sure not to change the factory address or the init code without first removing any existing data to prevent the two salt types from becoming commingled. There's also a *very* simple monitoring tool available if you run `$python3 analysis.py` in another tab.

This tool was originally built for use with [`Pr000xy`](https://github.com/0age/Pr000xy), including with [`Create2Factory`](https://github.com/0age/Pr000xy/blob/master/contracts/Create2Factory.sol) directly.

There is also a GPU search mode. To give it a try, include a fourth parameter specifying the device ID to use, and optionally a fifth and sixth parameter to filter returned results by a threshold based on leading zero bytes and total zero bytes, respectively. By way of example, to perform the same search as above, but using GPU device 2 and only returning results that create addresses with at least four leading zeroes or six total zeroes, use `$ cargo run --release $FACTORY $CALLER $INIT_CODE_HASH 2 4 6` (you'll also probably want to try tweaking the `WORK_SIZE` parameter in `src/lib.rs`).

## GPU backends and Apple Silicon

Two GPU backends are available, selected with `--backend <auto|opencl|metal>`:

- **Metal** (default on macOS): drives the kernel through Metal directly. On Apple Silicon this is roughly **10x faster** than going through Apple's deprecated OpenCL-over-Metal shim - same kernel, modern compiler and runtime.
- **OpenCL** (default elsewhere): the original backend. On Apple platforms it automatically uses a bit-interleaved 32-bit keccak kernel (Apple GPUs have no native 64-bit integer ALUs); override with `--kernel-bits 32|64`.

A GPU run also mines on all CPU cores by default; pass `--no-cpu` to keep the machine responsive (or to benchmark the GPU alone). On aarch64 CPUs with the ARMv8.4 SHA3 extension (all Apple M-series), the CPU path hashes two candidates at once in the 128-bit NEON registers via the `EOR3`/`RAX1`/`XAR`/`BCAX` instructions - about 2x the throughput of the scalar CRYPTOGAMS assembly path (which is used as the fallback and elsewhere). Both miners share the display and the output file.

Co-mining is not purely additive: the CPU competes with the GPU for memory bandwidth and thermal headroom, so the net gain on an M4 Max is about 15-20% over the GPU alone (the CPU's full standalone rate minus what the GPU gives up), in exchange for pinning every core.

Measured on an Apple M4 Max (14-core CPU, 40-core GPU); GPU figures are cool-state and vary with thermals:

| Configuration | Rate |
|---|---|
| CPU only, scalar (CRYPTOGAMS asm) | ~71 Mh/s |
| CPU only, 2-way SHA3 NEON | ~151 Mh/s |
| GPU, OpenCL backend, 64-bit kernel | ~63 Mh/s |
| GPU, OpenCL backend, bit-interleaved kernel | ~71 Mh/s |
| GPU, Metal backend, `--no-cpu` | ~740 Mh/s |
| GPU Metal + CPU (default) | ~800 Mh/s |

Running both, the GPU retains ~85% of its solo rate (shared memory bandwidth) while the CPU adds its own, for a ~15-20% net gain over the GPU alone.

At ~750 Mh/s, v4 hook flags alone (2^14) are instant, flags + a 4-character vanity prefix (2^30) averages under two seconds, flags + 6 characters (2^38) about six minutes, and flags + 8 characters (2^46) about a day.

## Pattern mining (prefixes, suffixes, and Uniswap v4 hooks)

Instead of scoring addresses by zero bytes, you can search for addresses matching an exact bit pattern. Three options may be combined (their fixed bits must agree), and each works on both the CPU and GPU paths:

- `--prefix <HEX>` — address must start with these hex characters (odd lengths allowed)
- `--suffix <HEX>` — address must end with these hex characters (odd lengths allowed)
- `--hook-flags <FLAGS>` — Uniswap v4 hook mining: requires `address & 0x3fff == FLAGS`, an exact match on all 14 permission bits (the same check as v4-periphery's `HookMiner`)

When any pattern option is given, the zero-byte thresholds default to disabled; passing them explicitly ANDs them with the pattern (e.g. `--hook-flags 0xC0` plus a leading-zeroes threshold of 4).

### Uniswap v4 hook flags

Hook addresses encode their permissions in the lowest 14 bits of the address (`Hooks.ALL_HOOK_MASK`), and `BaseHook` validates an exact match on deployment. Flag values from v4-core `Hooks.sol`:

| Flag | Bit | Value |
|------|-----|-------|
| `BEFORE_INITIALIZE_FLAG` | 13 | `0x2000` |
| `AFTER_INITIALIZE_FLAG` | 12 | `0x1000` |
| `BEFORE_ADD_LIQUIDITY_FLAG` | 11 | `0x0800` |
| `AFTER_ADD_LIQUIDITY_FLAG` | 10 | `0x0400` |
| `BEFORE_REMOVE_LIQUIDITY_FLAG` | 9 | `0x0200` |
| `AFTER_REMOVE_LIQUIDITY_FLAG` | 8 | `0x0100` |
| `BEFORE_SWAP_FLAG` | 7 | `0x0080` |
| `AFTER_SWAP_FLAG` | 6 | `0x0040` |
| `BEFORE_DONATE_FLAG` | 5 | `0x0020` |
| `AFTER_DONATE_FLAG` | 4 | `0x0010` |
| `BEFORE_SWAP_RETURNS_DELTA_FLAG` | 3 | `0x0008` |
| `AFTER_SWAP_RETURNS_DELTA_FLAG` | 2 | `0x0004` |
| `AFTER_ADD_LIQUIDITY_RETURNS_DELTA_FLAG` | 1 | `0x0002` |
| `AFTER_REMOVE_LIQUIDITY_RETURNS_DELTA_FLAG` | 0 | `0x0001` |

OR together the flags your hook's `getHookPermissions()` declares. For example, a hook using `beforeSwap` and `afterSwap` needs `0x0080 | 0x0040 = 0x00C0`.

### Example: mining a v4 hook address with a vanity prefix

When deploying via `forge script` with `new MyHook{salt: salt}(...)`, contracts are deployed through the CREATE2 Deployer Proxy at `0x4e59b44847b379578588920cA78FbF26c0B4956C`, which forwards the salt as-is — so use it as the factory and the zero address as the caller:

```sh
$ export FACTORY="0x4e59b44847b379578588920cA78FbF26c0B4956C"
$ export CALLER="0x0000000000000000000000000000000000000000"
$ export INIT_CODE_HASH="<keccak256 of your hook creation code ++ abi-encoded constructor args>"
$ cargo run --release -- $FACTORY $CALLER $INIT_CODE_HASH --hook-flags 0x00C0 --prefix c0ffee
```

The 14 flag bits alone cost ~2^14 attempts (instant); each additional vanity prefix character multiplies difficulty by 16. For long prefixes, add a GPU device (see the backends section below). The init code hash covers the constructor arguments too: `keccak256(abi.encodePacked(type(MyHook).creationCode, abi.encode(...)))`. Verify a mined salt before use:

```sh
$ cast create2 --deployer $FACTORY --salt <salt> --init-code-hash $INIT_CODE_HASH
```

PRs welcome!
