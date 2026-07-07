#![warn(unused_crate_dependencies, unreachable_pub)]
#![deny(unused_must_use, rust_2018_idioms)]

use alloy_primitives::{hex, Address, FixedBytes};
use clap::Parser;
use console::{style, Style, Term};
use fs4::FileExt;
use keccak_asm::{Digest, Keccak256};
use ocl::{Buffer, Context, Device, MemFlags, Platform, ProQue, Program, Queue};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::error::Error;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use terminal_size::{terminal_size, Height};

#[cfg(target_arch = "aarch64")]
mod keccak_x2;
#[cfg(target_os = "macos")]
mod metal_gpu;
mod reward;
pub use reward::Reward;

// workset size (tweak this!)
const WORK_SIZE: u32 = 0x4000000; // max. 0x15400000 to abs. max 0xffffffff

const CONTROL_CHARACTER: u8 = 0xff;
const MAX_INCREMENTER: u64 = 0xffffffffffff;

static KERNEL_SRC: &str = include_str!("./kernels/keccak256.cl");
static KERNEL_BI_CORE: &str = include_str!("./kernels/keccak_bi_core.cl");
static KERNEL_SRC_32: &str = include_str!("./kernels/keccak256_32.cl");
static KERNEL_SRC_MSL: &str = include_str!("./kernels/keccak256_32.metal");

/// Which GPU kernel to generate: the original 64-bit OpenCL kernel, the
/// bit-interleaved 32-bit OpenCL kernel, or the Metal (MSL) port of the
/// bit-interleaved kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KernelFlavor {
    OpenCl64,
    OpenCl32,
    Metal,
}

/// Keccak-f[1600] round constants. Only the first 23 are used by the kernels;
/// the final round is computed partially and skips iota.
const KECCAK_RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// Split a 64-bit keccak lane into its even bits and odd bits, each packed
/// into a 32-bit word. In this representation a 64-bit rotation costs only
/// one or two native 32-bit rotations, which is what makes keccak fast on
/// GPUs without 64-bit integer ALUs (e.g. Apple Silicon).
fn bit_interleave(lane: u64) -> (u32, u32) {
    let mut even = 0u32;
    let mut odd = 0u32;
    for i in 0..32 {
        even |= (((lane >> (2 * i)) & 1) as u32) << i;
        odd |= (((lane >> (2 * i + 1)) & 1) as u32) << i;
    }
    (even, odd)
}

/// Search for CREATE2 salts that produce gas-efficient or pattern-matching
/// contract addresses.
///
/// By default, addresses are scored by leading/total zero bytes (the classic
/// create2crunch behavior). Alternatively, provide --prefix, --suffix, and/or
/// --hook-flags to search for addresses matching an exact bit pattern - for
/// example, Uniswap v4 hook addresses, which encode their permissions in the
/// lowest 14 bits of the address and may also carry a vanity prefix.
#[derive(Parser, Debug)]
#[command(name = "create2crunch", version)]
pub struct Args {
    /// Address of the contract that will call CREATE2 (the factory)
    #[arg(value_parser = parse_fixed_bytes::<20>)]
    pub factory: [u8; 20],

    /// Address of the caller of the factory, for factories with frontrunning
    /// protection (use the zero address if not applicable)
    #[arg(value_parser = parse_fixed_bytes::<20>)]
    pub caller: [u8; 20],

    /// Keccak-256 hash of the initialization code of the contract to deploy
    #[arg(value_parser = parse_fixed_bytes::<32>)]
    pub init_code_hash: [u8; 32],

    /// OpenCL GPU device to use (255 = CPU)
    #[arg(default_value_t = 255)]
    pub gpu_device: u8,

    /// Leading zero-bytes threshold (defaults to 3, or 0 when a pattern is
    /// given; in pattern mode a non-zero value is ANDed with the pattern)
    pub leading_zeroes: Option<u8>,

    /// Total zero-bytes threshold (defaults to 5, or disabled when a pattern
    /// is given; in pattern mode a value of 0..=20 is ANDed with the pattern)
    pub total_zeroes: Option<u8>,

    /// Require the address to start with this hex string (up to 40 hex
    /// characters, odd lengths allowed)
    #[arg(long)]
    pub prefix: Option<String>,

    /// Require the address to end with this hex string (up to 40 hex
    /// characters, odd lengths allowed)
    #[arg(long)]
    pub suffix: Option<String>,

    /// Uniswap v4 hook permission flags: require address & 0x3fff == FLAGS
    /// (exact match on all 14 flag bits, hex or decimal)
    #[arg(long, value_name = "FLAGS", value_parser = parse_hook_flags)]
    pub hook_flags: Option<u16>,

    /// Also mine on all CPU cores while the GPU mines (no effect in CPU-only
    /// mode)
    #[arg(long)]
    pub cpu: bool,

    /// GPU kernel word size: 32 uses a bit-interleaved keccak for GPUs
    /// without native 64-bit integer ALUs (auto-selected on Apple; OpenCL
    /// backend only - Metal always uses the bit-interleaved kernel)
    #[arg(long, value_parser = parse_kernel_bits)]
    pub kernel_bits: Option<u8>,

    /// GPU backend to use
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    pub backend: Backend,
}

/// GPU compute backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Backend {
    /// Metal on macOS, OpenCL elsewhere
    Auto,
    /// OpenCL
    Opencl,
    /// Native Metal (macOS only)
    Metal,
}

fn parse_kernel_bits(s: &str) -> Result<u8, String> {
    match s {
        "32" => Ok(32),
        "64" => Ok(64),
        _ => Err("kernel bits must be 32 or 64".into()),
    }
}

fn parse_fixed_bytes<const N: usize>(s: &str) -> Result<[u8; N], String> {
    let bytes = hex::decode(s).map_err(|e| e.to_string())?;
    bytes
        .try_into()
        .map_err(|_| format!("expected {N} bytes ({} hex characters)", N * 2))
}

fn parse_hook_flags(s: &str) -> Result<u16, String> {
    let flags = match s.strip_prefix("0x") {
        Some(hex_str) => u16::from_str_radix(hex_str, 16),
        None => s.parse::<u16>(),
    }
    .map_err(|e| e.to_string())?;
    if flags > HOOK_FLAG_MASK {
        return Err(format!(
            "hook flags must fit in 14 bits (max 0x{HOOK_FLAG_MASK:x})"
        ));
    }
    Ok(flags)
}

/// Parse a hex string (optionally 0x-prefixed, odd lengths allowed) into
/// individual nibble values.
fn parse_nibbles(s: &str, what: &str) -> Result<Vec<u8>, String> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    if s.is_empty() {
        return Err(format!("{what} pattern is empty"));
    }
    if s.len() > 40 {
        return Err(format!("{what} pattern is longer than 40 hex characters"));
    }
    s.chars()
        .map(|c| {
            c.to_digit(16)
                .map(|d| d as u8)
                .ok_or_else(|| format!("{what} pattern contains non-hex character {c:?}"))
        })
        .collect()
}

/// Uniswap v4 `Hooks.ALL_HOOK_MASK`: hook permissions occupy the lowest 14
/// bits of the hook address and must match the declared permissions exactly.
pub const HOOK_FLAG_MASK: u16 = (1 << 14) - 1;

/// A bit-level constraint over a 20-byte address: an address matches when
/// `address & mask == value` for every byte. Built from vanity prefixes,
/// suffixes, and/or Uniswap v4 hook permission flags, which may be combined
/// as long as their fixed bits agree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pattern {
    mask: [u8; 20],
    value: [u8; 20],
}

impl Pattern {
    /// Constrain the address to start with the given nibbles.
    pub fn from_prefix(nibbles: &[u8]) -> Self {
        let mut pattern = Self::default();
        for (i, &nibble) in nibbles.iter().enumerate() {
            pattern.set_nibble(i, nibble);
        }
        pattern
    }

    /// Constrain the address to end with the given nibbles.
    pub fn from_suffix(nibbles: &[u8]) -> Self {
        let mut pattern = Self::default();
        for (i, &nibble) in nibbles.iter().enumerate() {
            pattern.set_nibble(40 - nibbles.len() + i, nibble);
        }
        pattern
    }

    /// Constrain the lowest 14 bits of the address to exactly equal the given
    /// Uniswap v4 hook permission flags.
    pub fn from_hook_flags(flags: u16) -> Self {
        let mut pattern = Self::default();
        pattern.mask[18] = (HOOK_FLAG_MASK >> 8) as u8;
        pattern.mask[19] = 0xff;
        pattern.value[18] = (flags >> 8) as u8;
        pattern.value[19] = (flags & 0xff) as u8;
        pattern
    }

    fn set_nibble(&mut self, position: usize, nibble: u8) {
        let byte = position / 2;
        if position % 2 == 0 {
            self.mask[byte] |= 0xf0;
            self.value[byte] |= nibble << 4;
        } else {
            self.mask[byte] |= 0x0f;
            self.value[byte] |= nibble;
        }
    }

    /// Combine two patterns, requiring agreement on any overlapping bits.
    pub fn merge(self, other: Self) -> Result<Self, String> {
        let mut merged = Self::default();
        for i in 0..20 {
            let overlap = self.mask[i] & other.mask[i];
            if (self.value[i] ^ other.value[i]) & overlap != 0 {
                return Err(format!(
                    "conflicting patterns: prefix/suffix/hook-flags disagree on address byte {i}"
                ));
            }
            merged.mask[i] = self.mask[i] | other.mask[i];
            merged.value[i] = self.value[i] | other.value[i];
        }
        Ok(merged)
    }

    /// Whether the given 20-byte address satisfies this pattern.
    pub fn matches(&self, address: &[u8]) -> bool {
        address
            .iter()
            .zip(&self.mask)
            .zip(&self.value)
            .all(|((&byte, &mask), &value)| byte & mask == value)
    }

    /// Number of constrained bits; a random address matches with probability
    /// 2^-bits().
    pub fn bits(&self) -> u32 {
        self.mask.iter().map(|m| m.count_ones()).sum()
    }
}

/// Renders the pattern as an address template: fixed nibbles as hex digits,
/// free nibbles as `x`, and partially-constrained nibbles (e.g. the top two
/// bits of the 14-bit hook flags) as `?`.
impl std::fmt::Display for Pattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x")?;
        for position in 0..40 {
            let byte = position / 2;
            let (mask, value) = if position % 2 == 0 {
                (self.mask[byte] >> 4, self.value[byte] >> 4)
            } else {
                (self.mask[byte] & 0xf, self.value[byte] & 0xf)
            };
            match mask {
                0x0 => write!(f, "x")?,
                0xf => write!(f, "{value:x}")?,
                _ => write!(f, "?")?,
            }
        }
        Ok(())
    }
}

/// The resolved search configuration: the CREATE2 inputs plus either the
/// classic zero-byte thresholds or a bit pattern to match.
#[derive(Clone)]
pub struct Config {
    pub factory_address: [u8; 20],
    pub calling_address: [u8; 20],
    pub init_code_hash: [u8; 32],
    pub gpu_device: u8,
    pub leading_zeroes_threshold: u8,
    pub total_zeroes_threshold: u8,
    pub pattern: Option<Pattern>,
    pub cpu: bool,
    pub kernel_bits: Option<u8>,
    pub backend: Backend,
}

impl Config {
    /// Parse the process arguments (exiting with help/usage on parse errors)
    /// and validate them into a Config.
    pub fn parse_args() -> Result<Self, String> {
        Self::new(Args::parse())
    }

    /// Validate the provided arguments and construct the Config struct.
    pub fn new(args: Args) -> Result<Self, String> {
        let mut pattern: Option<Pattern> = None;
        let mut add_pattern = |new: Pattern| -> Result<(), String> {
            pattern = Some(match pattern.take() {
                Some(existing) => existing.merge(new)?,
                None => new,
            });
            Ok(())
        };

        if let Some(prefix) = &args.prefix {
            add_pattern(Pattern::from_prefix(&parse_nibbles(prefix, "prefix")?))?;
        }
        if let Some(suffix) = &args.suffix {
            add_pattern(Pattern::from_suffix(&parse_nibbles(suffix, "suffix")?))?;
        }
        if let Some(flags) = args.hook_flags {
            add_pattern(Pattern::from_hook_flags(flags))?;
        }

        // in pattern mode the zero-byte thresholds default to disabled;
        // otherwise keep the classic defaults of 3 leading / 5 total
        let (leading_default, total_default) = if pattern.is_some() { (0, 255) } else { (3, 5) };
        let leading_zeroes_threshold = args.leading_zeroes.unwrap_or(leading_default);
        let total_zeroes_threshold = args.total_zeroes.unwrap_or(total_default);

        if leading_zeroes_threshold > 20 {
            return Err(
                "invalid value for leading zeroes threshold argument. (valid: 0..=20)".into(),
            );
        }
        if total_zeroes_threshold > 20 && total_zeroes_threshold != 255 {
            return Err(
                "invalid value for total zeroes threshold argument. (valid: 0..=20 | 255)".into(),
            );
        }

        Ok(Self {
            factory_address: args.factory,
            calling_address: args.caller,
            init_code_hash: args.init_code_hash,
            gpu_device: args.gpu_device,
            leading_zeroes_threshold,
            total_zeroes_threshold,
            pattern,
            cpu: args.cpu,
            kernel_bits: args.kernel_bits,
            backend: args.backend,
        })
    }
}

/// Dispatch mining according to the configuration: CPU-only, GPU-only
/// (OpenCL or Metal), or GPU with CPU mining concurrently on a background
/// thread (--cpu).
pub fn run(config: Config) -> Result<(), Box<dyn Error>> {
    if config.gpu_device == 255 {
        return cpu(config, None);
    }

    let use_metal = match config.backend {
        Backend::Metal => true,
        Backend::Opencl => false,
        Backend::Auto => cfg!(target_os = "macos"),
    };

    // when also mining on the CPU, share one progress tracker between both
    // miners so the GPU's display reflects the combined effort; the tracker
    // is populated with backend/device details by the GPU backend below
    let shared: Option<Arc<Progress>> = if config.cpu {
        // leave one core free for the GPU host thread
        let threads = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();

        let progress = Arc::new(Progress::new(&config));
        let cpu_config = config.clone();
        let cpu_progress = progress.clone();
        std::thread::spawn(move || {
            if let Err(e) = cpu(cpu_config, Some(cpu_progress)) {
                eprintln!("CPU mining error: {e}");
            }
        });
        Some(progress)
    } else {
        None
    };

    if use_metal {
        #[cfg(target_os = "macos")]
        return metal_gpu::gpu(config, shared);
        #[cfg(not(target_os = "macos"))]
        return Err("the Metal backend is only available on macOS (use --backend opencl)".into());
    }

    gpu(config, shared)?;
    Ok(())
}

/// Shared, thread-safe view of a mining run's progress: total hashes tried,
/// the solutions found so far, and a static summary of the run configuration
/// for the status display. One instance is shared between concurrent miners
/// (GPU + `--cpu`) so a single display can report their combined effort.
pub struct Progress {
    hashes: AtomicU64,
    found: Mutex<Vec<String>>,
    start: Instant,
    /// backend and device names, set once the compute device is opened
    backend: OnceLock<(String, String)>,
    /// static configuration rows (label, value) rendered in the header
    config_rows: Vec<(&'static str, String)>,
    /// the search target/criteria description
    target: String,
    /// difficulty in bits, i.e. -log2(probability a random address qualifies)
    difficulty_bits: f64,
}

impl Progress {
    fn new(config: &Config) -> Self {
        let mut config_rows = vec![
            (
                "factory",
                Address::from(config.factory_address).to_checksum(None),
            ),
            (
                "caller",
                Address::from(config.calling_address).to_checksum(None),
            ),
            (
                "init hash",
                format!("0x{}", hex::encode(config.init_code_hash)),
            ),
        ];
        if config.cpu {
            config_rows.push(("+ cpu", cpu_kernel_label().to_string()));
        }

        let (target, difficulty_bits) = match &config.pattern {
            Some(pattern) => (format!("{pattern}"), pattern.bits() as f64),
            None => {
                let l = config.leading_zeroes_threshold;
                let t = config.total_zeroes_threshold;
                let criteria = if t <= 20 {
                    format!("\u{2265}{l} leading or \u{2265}{t} total zero bytes")
                } else {
                    format!("\u{2265}{l} leading zero bytes")
                };
                (criteria, zero_byte_difficulty_bits(l, t))
            }
        };

        Self {
            hashes: AtomicU64::new(0),
            found: Mutex::new(Vec::new()),
            start: Instant::now(),
            backend: OnceLock::new(),
            config_rows,
            target,
            difficulty_bits,
        }
    }

    /// Record `n` completed hash attempts.
    fn add_hashes(&self, n: u64) {
        self.hashes.fetch_add(n, Ordering::Relaxed);
    }

    /// Record a found solution (already formatted for both file and display).
    fn push_found(&self, entry: String) {
        self.found.lock().unwrap().push(entry);
    }

    /// Name the backend and device once the compute device is opened.
    fn set_backend(&self, backend: impl Into<String>, device: impl Into<String>) {
        let _ = self.backend.set((backend.into(), device.into()));
    }
}

/// Owns the terminal and paints [`Progress`] as a live status screen. A run
/// has exactly one renderer (the GPU host thread, or the CPU reporter
/// thread), so the windowed-rate state it keeps needs no synchronization.
pub(crate) struct Renderer {
    term: Term,
    progress: Arc<Progress>,
    last_paint: Option<Instant>,
    window_start: Instant,
    window_hashes: u64,
    rate: f64,
}

impl Renderer {
    fn new(progress: Arc<Progress>) -> Self {
        Self {
            term: Term::stdout(),
            progress,
            last_paint: None,
            window_start: Instant::now(),
            window_hashes: 0,
            rate: 0.0,
        }
    }

    /// Repaint the screen if at least ~1s has elapsed since the last paint.
    fn tick(&mut self) -> std::io::Result<()> {
        let now = Instant::now();
        if let Some(last) = self.last_paint {
            if now.duration_since(last).as_secs_f64() < 1.0 {
                return Ok(());
            }
        }
        self.last_paint = Some(now);
        self.paint(now)
    }

    fn paint(&mut self, now: Instant) -> std::io::Result<()> {
        let p = &self.progress;
        let hashes = p.hashes.load(Ordering::Relaxed);
        let elapsed = now.duration_since(p.start).as_secs_f64();

        // windowed hash rate, refreshed each paint so it tracks current
        // throughput rather than the lifetime average
        let window_secs = now.duration_since(self.window_start).as_secs_f64();
        if window_secs > 0.0 {
            self.rate = (hashes - self.window_hashes) as f64 / window_secs;
        }
        self.window_start = now;
        self.window_hashes = hashes;
        let rate = self.rate;

        // expected work per match and its timing; difficulty_bits can exceed
        // f64's integer range but 2^bits stays representable up to ~1024 bits
        let expected_hashes = 2f64.powf(p.difficulty_bits);
        let found = p.found.lock().unwrap();
        let dim = Style::new().dim();
        let label = |s: &str| format!("{:>11}", style(s).dim());

        let mut out = String::new();
        let rule = "\u{2500}".repeat(64);
        let _ = writeln!(out, "{}", style(format!("create2crunch {rule}")).cyan());

        // configuration block
        if let Some((backend, device)) = p.backend.get() {
            let _ = writeln!(
                out,
                "{} {} {}",
                label("backend"),
                backend,
                dim.apply_to(format!("\u{00b7} {device}"))
            );
        }
        for (name, value) in &p.config_rows {
            let _ = writeln!(out, "{} {}", label(name), value);
        }
        let _ = writeln!(
            out,
            "{} {}   {}",
            label("target"),
            style(&p.target).green(),
            dim.apply_to(format!("1 in 2^{:.0}", p.difficulty_bits.round()))
        );
        let _ = writeln!(out, "{}", dim.apply_to(&rule));

        // live statistics
        let _ = writeln!(
            out,
            "{} {:<22}{} {}",
            label("elapsed"),
            fmt_duration(elapsed),
            label("hashes"),
            fmt_count(hashes),
        );
        let _ = writeln!(
            out,
            "{} {:<22}{} {}",
            label("rate"),
            style(format!("{:.1} Mh/s", rate / 1e6)).bold(),
            label("found"),
            style(found.len().to_string()).bold(),
        );

        // expected time to the next match, and how "due" the search is
        let (eta, prob) = if rate > 0.0 {
            let secs = expected_hashes / rate;
            // probability of at least one match given hashes tried so far
            let prob = 1.0 - (-(hashes as f64) / expected_hashes).exp();
            (fmt_duration(secs), format!("{:.0}%", prob * 100.0))
        } else {
            ("\u{2014}".to_string(), "\u{2014}".to_string())
        };
        let _ = writeln!(
            out,
            "{} {:<22}{} {}",
            label("avg/match"),
            eta,
            label("P(\u{2265}1 hit)"),
            prob,
        );

        // recent finds, filling the remaining terminal height
        let height = terminal_size().map(|(_w, Height(h))| h).unwrap_or(24) as usize;
        let used = p.config_rows.len() + p.backend.get().is_some() as usize + 6;
        let rows = height.saturating_sub(used + 1).max(1);
        if !found.is_empty() {
            let _ = writeln!(
                out,
                "{}",
                dim.apply_to(format!("recent {}", "\u{2500}".repeat(57)))
            );
            let start = found.len().saturating_sub(rows);
            for entry in &found[start..] {
                let _ = writeln!(out, "{entry}");
            }
        }

        self.term.clear_screen()?;
        self.term.write_str(&out)
    }
}

/// The label for the CPU hashing path in use on this machine.
fn cpu_kernel_label() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("sha3")
        && std::env::var_os("CREATE2CRUNCH_FORCE_SCALAR").is_none()
    {
        return "2-way SHA3 NEON";
    }
    "scalar (CRYPTOGAMS asm)"
}

/// Approximate difficulty (in bits) of the classic zero-byte criteria
/// "\u{2265}L leading OR \u{2265}T total zero bytes": -log2 of the probability
/// that a uniformly random 20-byte address qualifies.
fn zero_byte_difficulty_bits(leading: u8, total: u8) -> f64 {
    // P(first L bytes zero) = 256^-L
    let p_leading = 256f64.powi(-(leading as i32));
    // P(>=T of 20 bytes zero), binomial tail with p = 1/256
    let p_total = if total > 20 {
        0.0
    } else {
        let p = 1.0f64 / 256.0;
        let mut tail = 0.0;
        for k in total as u32..=20 {
            let mut term = binomial(20, k) as f64;
            term *= p.powi(k as i32);
            term *= (1.0 - p).powi((20 - k) as i32);
            tail += term;
        }
        tail
    };
    // union bound; the two events overlap negligibly at these probabilities
    let p = (p_leading + p_total).min(1.0);
    if p <= 0.0 {
        f64::INFINITY
    } else {
        -p.log2()
    }
}

fn binomial(n: u32, k: u32) -> u64 {
    let k = k.min(n - k);
    let mut result: u64 = 1;
    for i in 0..k {
        result = result * (n - i) as u64 / (i + 1) as u64;
    }
    result
}

/// Format a byte/attempt count with an SI-style suffix (e.g. `1.23 billion`).
fn fmt_count(n: u64) -> String {
    const UNITS: [(f64, &str); 4] = [
        (1e12, "trillion"),
        (1e9, "billion"),
        (1e6, "million"),
        (1e3, "thousand"),
    ];
    let f = n as f64;
    for (scale, name) in UNITS {
        if f >= scale {
            return format!("{:.2} {name}", f / scale);
        }
    }
    n.to_string()
}

/// Format a duration in seconds as a compact human string (e.g. `2h 3m`,
/// `6m 12s`, `45s`), or an eternity marker for absurdly long estimates.
fn fmt_duration(secs: f64) -> String {
    if !secs.is_finite() || secs >= 315_360_000.0 {
        // >= 10 years: not worth a precise figure
        return "> 10 years".to_string();
    }
    let secs = secs.max(0.0) as u64;
    let (d, h, m, s) = (secs / 86400, secs / 3600 % 24, secs / 60 % 60, secs % 60);
    if d > 0 {
        format!("{d}d {h}h")
    } else if h > 0 {
        format!("{h}h {m}m")
    } else if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    }
}

/// Given a Config object with a factory address, a caller address, and a
/// keccak-256 hash of the contract initialization code, search for salts that
/// will enable the factory contract to deploy a contract to a gas-efficient
/// address via CREATE2.
///
/// The 32-byte salt is constructed as follows:
///   - the 20-byte calling address (to prevent frontrunning)
///   - a random 6-byte segment (to prevent collisions with other runs)
///   - a 6-byte nonce segment (incrementally stepped through during the run)
///
/// When a salt that will result in the creation of a gas-efficient contract
/// address is found, it will be appended to `efficient_addresses.txt` along
/// with the resultant address and the "value" (i.e. approximate rarity) of the
/// resultant address.
pub fn cpu(config: Config, progress: Option<Arc<Progress>>) -> Result<(), Box<dyn Error>> {
    // (create if necessary) and open a file where found salts will be written
    let file = output_file();

    // create object for computing rewards (relative rarity) for a given address
    let rewards = Reward::new();

    // when running standalone, own the progress tracker and paint the status
    // screen from a background thread; when co-mining, the GPU backend owns
    // both and this call only feeds hashes/finds into the shared tracker
    let solo = progress.is_none();
    let progress = progress.unwrap_or_else(|| {
        let p = Arc::new(Progress::new(&config));
        let threads = rayon::current_num_threads();
        p.set_backend(
            "CPU",
            format!("{threads} threads \u{00b7} {}", cpu_kernel_label()),
        );
        p
    });
    if solo {
        let reporter = progress.clone();
        std::thread::spawn(move || {
            let mut renderer = Renderer::new(reporter);
            loop {
                // short sleep so the thread stays responsive; tick() itself
                // throttles the actual repaint to ~1s
                std::thread::sleep(std::time::Duration::from_millis(250));
                let _ = renderer.tick();
            }
        });
    }

    // on CPUs with the ARMv8.4 SHA3 extension, hash two candidates per core
    // in the 128-bit NEON registers (CREATE2CRUNCH_FORCE_SCALAR overrides,
    // mainly for benchmarking)
    #[cfg(target_arch = "aarch64")]
    let use_x2 = std::arch::is_aarch64_feature_detected!("sha3")
        && std::env::var_os("CREATE2CRUNCH_FORCE_SCALAR").is_none();

    // hashes counted (and the status refreshed) once per chunk to keep the
    // shared atomic off the innermost loop
    const CHUNK: u64 = 1 << 16;

    // begin searching for addresses
    loop {
        // message: 0xff ++ factory ++ caller ++ salt_random_segment (47 bytes)
        // ++ 6-byte nonce ++ init code hash (85 bytes total)
        let mut template = [0u8; 85];
        template[0] = CONTROL_CHARACTER;
        template[1..21].copy_from_slice(&config.factory_address);
        template[21..41].copy_from_slice(&config.calling_address);
        template[41..47].copy_from_slice(&FixedBytes::<6>::random()[..]);
        template[53..].copy_from_slice(&config.init_code_hash);

        // score one candidate's address and record it if it qualifies;
        // shared by the scalar and 2-way paths. The cheapest rejection for
        // the active mode runs first so the hash dominates the hot loop.
        let handle_candidate = |salt: u64, address_bytes: &[u8]| {
            let reward_amount = match &config.pattern {
                Some(pattern) => {
                    // pattern mode: the masked compare rejects almost every
                    // candidate on its first differing byte, so check it
                    // before the zero-byte census
                    if !pattern.matches(address_bytes) {
                        return;
                    }
                    let (leading, total) = count_zero_bytes(address_bytes);
                    if (leading.min(20) as u8) < config.leading_zeroes_threshold {
                        return;
                    }
                    if config.total_zeroes_threshold <= 20
                        && (total as u8) < config.total_zeroes_threshold
                    {
                        return;
                    }
                    rewards.get(&(leading * 20 + total))
                }
                None => {
                    // zero-byte mode: reject on too few total zeros first
                    let (leading, total) = count_zero_bytes(address_bytes);
                    if total < 3 {
                        return;
                    }

                    // only proceed if an efficient address has been found
                    let reward_amount = rewards.get(&(leading * 20 + total));
                    if reward_amount.is_none() {
                        return;
                    }
                    reward_amount
                }
            };

            let address = <&Address>::try_from(address_bytes).unwrap();

            // get the full salt used to create the address
            let salt_bytes = salt.to_le_bytes();
            let full_salt = format!(
                "0x{}{}",
                hex::encode(&template[21..47]),
                hex::encode(&salt_bytes[..6])
            );

            // display the salt and the address.
            let output = format!(
                "{full_salt} => {address} => {}",
                reward_amount.unwrap_or("0")
            );
            progress.push_found(output.clone());

            // create a lock on the file before writing
            file.lock_exclusive().expect("Couldn't lock file.");

            // write the result to file
            writeln!(&file, "{output}").expect("Couldn't write to `efficient_addresses.txt` file.");

            // release the file lock
            file.unlock().expect("Couldn't unlock file.")
        };

        // process the nonce space in chunks so the hash counter is updated
        // in bulk rather than per candidate
        let run_chunk = |chunk: u64| {
            let base = chunk * CHUNK;

            #[cfg(target_arch = "aarch64")]
            if use_x2 {
                let lanes = sponge_lanes(&template);
                for pair in 0..CHUNK / 2 {
                    let (salt_a, salt_b) = (base + pair * 2, base + pair * 2 + 1);
                    // SAFETY: gated on runtime detection of the sha3 feature
                    let (address_a, address_b) =
                        unsafe { keccak_x2::address_pair(&lanes, salt_a, salt_b) };
                    handle_candidate(salt_a, &address_a);
                    handle_candidate(salt_b, &address_b);
                }
                progress.add_hashes(CHUNK);
                return;
            }

            for k in 0..CHUNK {
                let salt = base + k;
                // splice the nonce into the message and hash it
                let mut message = template;
                message[47..53].copy_from_slice(&salt.to_le_bytes()[..6]);
                let res = Keccak256::digest(message);
                handle_candidate(salt, &res[12..]);
            }
            progress.add_hashes(CHUNK);
        };

        (0..MAX_INCREMENTER / CHUNK)
            .into_par_iter() // parallelization
            .for_each(run_chunk);
    }
}

/// Count leading and total zero bytes of a 20-byte address. `leading` is 21
/// for the all-zero address (matching the original sentinel).
fn count_zero_bytes(address: &[u8]) -> (usize, usize) {
    let mut total = 0;
    let mut leading = 21;
    for (i, &b) in address.iter().enumerate() {
        if b == 0 {
            total += 1;
        } else if leading == 21 {
            leading = i;
        }
    }
    (leading, total)
}

/// The 17 rate lanes of the keccak sponge for the padded 85-byte miner
/// message (capacity lanes 17..=24 are zero).
#[cfg(target_arch = "aarch64")]
fn sponge_lanes(template: &[u8; 85]) -> [u64; 17] {
    let mut block = [0u8; 136];
    block[..85].copy_from_slice(template);
    block[85] = 0x01; // keccak-256 padding start
    block[135] = 0x80; // keccak-256 padding end
    let mut lanes = [0u64; 17];
    for (j, lane) in lanes.iter_mut().enumerate() {
        *lane = u64::from_le_bytes(block[8 * j..8 * j + 8].try_into().unwrap());
    }
    lanes
}

/// Given a Config object with a factory address, a caller address, a keccak-256
/// hash of the contract initialization code, and a device ID, search for salts
/// using OpenCL that will enable the factory contract to deploy a contract to a
/// gas-efficient address via CREATE2. This method also takes threshold values
/// for both leading zero bytes and total zero bytes - any address that does not
/// meet or exceed the threshold will not be returned. Default threshold values
/// are three leading zeroes or five total zeroes.
///
/// The 32-byte salt is constructed as follows:
///   - the 20-byte calling address (to prevent frontrunning)
///   - a random 4-byte segment (to prevent collisions with other runs)
///   - a 4-byte segment unique to each work group running in parallel
///   - a 4-byte nonce segment (incrementally stepped through during the run)
///
/// When a salt that will result in the creation of a gas-efficient contract
/// address is found, it will be appended to `efficient_addresses.txt` along
/// with the resultant address and the "value" (i.e. approximate rarity) of the
/// resultant address.
///
/// This method is still highly experimental and could almost certainly use
/// further optimization - contributions are more than welcome!
pub fn gpu(config: Config, progress: Option<Arc<Progress>>) -> ocl::Result<()> {
    // (create if necessary) and open a file where found salts will be written
    let file = output_file();

    // create object for computing rewards (relative rarity) for a given address
    let rewards = Reward::new();

    // progress tracker (shared with a concurrent CPU miner when --cpu)
    let progress = progress.unwrap_or_else(|| Arc::new(Progress::new(&config)));

    // set up a platform to use
    let platform = Platform::new(ocl::core::default_platform()?);

    // Apple GPUs have no native 64-bit integer ALUs, so the bit-interleaved
    // 32-bit kernel is substantially faster there; allow manual override
    let use_32bit = match config.kernel_bits {
        Some(bits) => bits == 32,
        None => platform.name()?.contains("Apple"),
    };
    let flavor = if use_32bit {
        KernelFlavor::OpenCl32
    } else {
        KernelFlavor::OpenCl64
    };

    // set up the device to use
    let device = Device::by_idx_wrap(platform, config.gpu_device as usize)?;
    progress.set_backend(
        if use_32bit {
            "OpenCL (bit-interleaved 32-bit)"
        } else {
            "OpenCL (64-bit)"
        },
        device
            .name()
            .unwrap_or_else(|_| "unknown device".to_string()),
    );

    // set up the context to use
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    // set up the program to use
    let program = Program::builder()
        .devices(device)
        .src(mk_kernel_src(&config, flavor))
        .build(&context)?;

    // set up the queue to use
    let queue = Queue::new(&context, device, None)?;

    // set up the "proqueue" (or amalgamation of various elements) to use
    let ocl_pq = ProQue::new(context, queue, program, Some(WORK_SIZE));

    // create a random number generator
    let mut rng = thread_rng();

    // set up the terminal status display
    let mut renderer = Renderer::new(progress.clone());

    // the last work duration in milliseconds
    let mut work_duration_millis: u64 = 0;

    // begin searching for addresses
    loop {
        // construct the 4-byte message to hash, leaving last 8 of salt empty
        let salt = FixedBytes::<4>::random();

        // build a corresponding buffer for passing the message to the kernel
        let message_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_only())
            .len(4)
            .copy_host_slice(&salt[..])
            .build()?;

        // reset the nonce; for more uniformly distributed nonces, we shall
        // initialize it to a random value
        let mut nonce: [u32; 1] = rng.gen();

        // build a corresponding buffer for passing the nonce to the kernel
        let mut nonce_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_only())
            .len(1)
            .copy_host_slice(&nonce)
            .build()?;

        // establish a buffer for nonces that result in desired addresses
        let mut solutions: Vec<u64> = vec![0; 1];
        let solutions_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().write_only())
            .len(1)
            .copy_host_slice(&solutions)
            .build()?;

        // repeatedly enqueue kernel to search for new addresses
        loop {
            // build the kernel and define the type of each buffer
            let kern = ocl_pq
                .kernel_builder("hashMessage")
                .arg_named("message", None::<&Buffer<u8>>)
                .arg_named("nonce", None::<&Buffer<u32>>)
                .arg_named("solutions", None::<&Buffer<u64>>)
                .build()?;

            // set each buffer
            kern.set_arg("message", Some(&message_buffer))?;
            kern.set_arg("nonce", Some(&nonce_buffer))?;
            kern.set_arg("solutions", &solutions_buffer)?;

            // enqueue the kernel
            unsafe { kern.enq()? };

            // calculate the current time
            let mut now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();

            // count this work group and repaint the status (throttled to ~1s)
            progress.add_hashes(WORK_SIZE as u64);
            renderer.tick()?;

            // record the start time of the work
            let work_start_time_millis = now.as_secs() * 1000 + now.subsec_nanos() as u64 / 1000000;

            // sleep for 98% of the previous work duration to conserve CPU
            if work_duration_millis != 0 {
                std::thread::sleep(std::time::Duration::from_millis(
                    work_duration_millis * 980 / 1000,
                ));
            }

            // read the solutions from the device
            solutions_buffer.read(&mut solutions).enq()?;

            // record the end time of the work and compute how long the work took
            now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            work_duration_millis = (now.as_secs() * 1000 + now.subsec_nanos() as u64 / 1000000)
                - work_start_time_millis;

            // if at least one solution is found, end the loop
            if solutions[0] != 0 {
                break;
            }

            // if no solution has yet been found, increment the nonce
            nonce[0] += 1;

            // update the nonce buffer with the incremented nonce value
            nonce_buffer = Buffer::builder()
                .queue(ocl_pq.queue().clone())
                .flags(MemFlags::new().read_write())
                .len(1)
                .copy_host_slice(&nonce)
                .build()?;
        }

        // iterate over each solution, first converting to a fixed array
        for &solution in &solutions {
            record_solution(&config, &rewards, &file, &progress, &salt[..], solution);
        }
    }
}

/// Recompute the address for a GPU-found nonce, re-verify it on the host,
/// and record it to the found list and the output file. Shared between the
/// OpenCL and Metal backends.
pub(crate) fn record_solution(
    config: &Config,
    rewards: &Reward,
    file: &File,
    progress: &Progress,
    salt: &[u8],
    solution: u64,
) {
    if solution == 0 {
        return;
    }

    let solution = solution.to_le_bytes();

    let mut solution_message = [0; 85];
    solution_message[0] = CONTROL_CHARACTER;
    solution_message[1..21].copy_from_slice(&config.factory_address);
    solution_message[21..41].copy_from_slice(&config.calling_address);
    solution_message[41..45].copy_from_slice(salt);
    solution_message[45..53].copy_from_slice(&solution);
    solution_message[53..].copy_from_slice(&config.init_code_hash);

    // hash the payload and get the result
    let res = Keccak256::digest(solution_message);

    // get the address that results from the hash
    let address = <&Address>::try_from(&res[12..]).unwrap();

    // re-verify the pattern on the host before recording the result
    if let Some(pattern) = &config.pattern {
        if !pattern.matches(&res[12..]) {
            return;
        }
    }

    // count total and leading zero bytes
    let mut total = 0;
    let mut leading = 0;
    for (i, &b) in address.iter().enumerate() {
        if b == 0 {
            total += 1;
        } else if leading == 0 {
            // set leading on finding non-zero byte
            leading = i;
        }
    }

    let key = leading * 20 + total;
    let reward = rewards.get(&key).unwrap_or("0");
    let output = format!(
        "0x{}{}{} => {} => {}",
        hex::encode(config.calling_address),
        hex::encode(salt),
        hex::encode(solution),
        address,
        reward,
    );

    let show = format!("{output} ({leading} / {total})");
    progress.push_found(show);

    file.lock_exclusive().expect("Couldn't lock file.");

    writeln!(&*file, "{output}").expect("Couldn't write to `efficient_addresses.txt` file.");

    file.unlock().expect("Couldn't unlock file.");
}

#[track_caller]
fn output_file() -> File {
    OpenOptions::new()
        .append(true)
        .create(true)
        .read(true)
        .open("efficient_addresses.txt")
        .expect("Could not create or open `efficient_addresses.txt` file.")
}

/// Creates the GPU kernel source code (OpenCL C or MSL, per the flavor) by
/// populating the template with the values from the Config object.
fn mk_kernel_src(config: &Config, flavor: KernelFlavor) -> String {
    let mut src = String::with_capacity(8192 + KERNEL_SRC.len().max(KERNEL_BI_CORE.len()));

    let factory = config.factory_address.iter();
    let caller = config.calling_address.iter();
    let hash = config.init_code_hash.iter();
    let hash = hash.enumerate().map(|(i, x)| (i + 52, x));
    for (i, x) in factory.chain(caller).enumerate().chain(hash) {
        writeln!(src, "#define S_{} {}u", i + 1, x).unwrap();
    }
    let lz = config.leading_zeroes_threshold;
    writeln!(src, "#define LEADING_ZEROES {lz}").unwrap();
    let tz = config.total_zeroes_threshold;
    writeln!(src, "#define TOTAL_ZEROES {tz}").unwrap();

    if let Some(pattern) = &config.pattern {
        // emit one comparison per constrained byte; bytes with a zero mask
        // are unconstrained and skipped entirely
        let conditions = pattern
            .mask
            .iter()
            .zip(&pattern.value)
            .enumerate()
            .filter(|(_, (&mask, _))| mask != 0)
            .map(|(i, (&mask, &value))| {
                if mask == 0xff {
                    format!("((d)[{i}] == {value}u)")
                } else {
                    format!("(((d)[{i}] & {mask}u) == {value}u)")
                }
            })
            .collect::<Vec<_>>()
            .join(" && ");
        writeln!(src, "#define PATTERN 1").unwrap();
        writeln!(src, "#define hasPattern(d) ({conditions})").unwrap();
    }

    if flavor == KernelFlavor::OpenCl64 {
        src.push_str(KERNEL_SRC);
        return src;
    }

    // the sponge template with all compile-time-constant bytes in place;
    // bytes 41..53 (salt_random_segment and nonce) are set per work item
    let mut template = [0u8; 136];
    template[0] = CONTROL_CHARACTER;
    template[1..21].copy_from_slice(&config.factory_address);
    template[21..41].copy_from_slice(&config.calling_address);
    template[53..85].copy_from_slice(&config.init_code_hash);
    template[85] = 0x01; // keccak-256 padding start
    template[135] = 0x80; // keccak-256 padding end

    // lanes 5 and 6 hold the per-work-item bytes and are assembled in the
    // kernel; lanes 11..=15 and 17..=24 are zero
    for i in [0usize, 1, 2, 3, 4, 7, 8, 9, 10, 16] {
        let lane = u64::from_le_bytes(template[8 * i..8 * i + 8].try_into().unwrap());
        let (even, odd) = bit_interleave(lane);
        writeln!(src, "#define A{i}_E {even}u").unwrap();
        writeln!(src, "#define A{i}_O {odd}u").unwrap();
    }

    // pre-interleaved iota round constants (the partial 24th round needs
    // no iota, so only 23 are emitted)
    for (i, rc) in KECCAK_RC.iter().enumerate().take(23) {
        let (even, odd) = bit_interleave(*rc);
        writeln!(src, "#define RC{i}_E {even}u").unwrap();
        writeln!(src, "#define RC{i}_O {odd}u").unwrap();
    }

    // splice the shared bit-interleaved core into the per-API wrapper
    let wrapper = match flavor {
        KernelFlavor::OpenCl32 => KERNEL_SRC_32,
        KernelFlavor::Metal => KERNEL_SRC_MSL,
        KernelFlavor::OpenCl64 => unreachable!(),
    };
    let (prelude, entry) = wrapper
        .split_once("//__CORE__//")
        .expect("kernel wrapper is missing the //__CORE__// marker");
    src.push_str(prelude);
    src.push_str(KERNEL_BI_CORE);
    src.push_str(entry);

    src
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_args() -> Args {
        Args {
            factory: [0x11; 20],
            caller: [0x22; 20],
            init_code_hash: [0x33; 32],
            gpu_device: 255,
            leading_zeroes: None,
            total_zeroes: None,
            prefix: None,
            suffix: None,
            hook_flags: None,
            cpu: false,
            kernel_bits: None,
            backend: Backend::Auto,
        }
    }

    /// Mirrors the delta-swap networks in keccak256_32.cl so the kernel's
    /// interleaving is pinned against the reference implementation.
    fn delta_swap(x: u32, shift: u32, mask: u32) -> u32 {
        let t = (x ^ (x >> shift)) & mask;
        x ^ t ^ (t << shift)
    }

    fn kernel_unshuffle(mut x: u32) -> u32 {
        x = delta_swap(x, 1, 0x22222222);
        x = delta_swap(x, 2, 0x0c0c0c0c);
        x = delta_swap(x, 4, 0x00f000f0);
        x = delta_swap(x, 8, 0x0000ff00);
        x
    }

    fn kernel_shuffle(mut x: u32) -> u32 {
        x = delta_swap(x, 8, 0x0000ff00);
        x = delta_swap(x, 4, 0x00f000f0);
        x = delta_swap(x, 2, 0x0c0c0c0c);
        x = delta_swap(x, 1, 0x22222222);
        x
    }

    fn kernel_interleave(lane: u64) -> (u32, u32) {
        let lo = kernel_unshuffle(lane as u32);
        let hi = kernel_unshuffle((lane >> 32) as u32);
        ((lo & 0xffff) | (hi << 16), (lo >> 16) | (hi & 0xffff0000))
    }

    #[test]
    fn kernel_bit_interleaving_matches_reference() {
        // deterministic pseudorandom coverage (splitmix64)
        let mut x = 0x9e3779b97f4a7c15u64;
        let mut lanes = vec![0, 1, u64::MAX, 0x8000000080008008];
        for _ in 0..1000 {
            x = x.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            lanes.push(z ^ (z >> 31));
        }
        for lane in lanes {
            assert_eq!(
                kernel_interleave(lane),
                bit_interleave(lane),
                "lane {lane:#x}"
            );
            let (e, o) = bit_interleave(lane);
            // the kernel's digest de-interleaving must invert the interleave
            let lo = kernel_shuffle((e & 0xffff) | ((o & 0xffff) << 16));
            let hi = kernel_shuffle((e >> 16) | (o & 0xffff0000));
            assert_eq!(((hi as u64) << 32) | lo as u64, lane, "lane {lane:#x}");
        }
    }

    #[test]
    fn hook_flags_constrain_lowest_14_bits() {
        // BEFORE_SWAP (1 << 7) | AFTER_SWAP (1 << 6) | BEFORE_INITIALIZE (1 << 13)
        let flags = (1u16 << 7) | (1 << 6) | (1 << 13);
        let pattern = Pattern::from_hook_flags(flags);
        assert_eq!(pattern.bits(), 14);

        let mut address = [0u8; 20];
        address[18] = (flags >> 8) as u8;
        address[19] = (flags & 0xff) as u8;
        assert!(pattern.matches(&address));

        // the top two bits of byte 18 are outside the 14-bit mask
        address[18] |= 0xc0;
        assert!(pattern.matches(&address));

        // any flipped flag bit must reject
        address[19] ^= 1;
        assert!(!pattern.matches(&address));
    }

    #[test]
    fn prefix_and_suffix_handle_odd_nibble_counts() {
        let prefix = Pattern::from_prefix(&parse_nibbles("c0ffe", "prefix").unwrap());
        let suffix = Pattern::from_suffix(&parse_nibbles("abc", "suffix").unwrap());

        let mut address = [0u8; 20];
        address[0] = 0xc0;
        address[1] = 0xff;
        address[2] = 0xe7; // low nibble free
        address[18] = 0x0a; // high nibble free
        address[19] = 0xbc;
        assert!(prefix.matches(&address));
        assert!(suffix.matches(&address));

        address[2] = 0xd7;
        assert!(!prefix.matches(&address));
        address[2] = 0xe7;
        address[18] = 0x0b;
        assert!(!suffix.matches(&address));
    }

    #[test]
    fn merge_rejects_conflicting_bits_and_accepts_agreement() {
        let suffix = Pattern::from_suffix(&parse_nibbles("ff", "suffix").unwrap());
        let flags_conflicting = Pattern::from_hook_flags(0);
        assert!(suffix.merge(flags_conflicting).is_err());

        let flags_agreeing = Pattern::from_hook_flags(0xff);
        let merged = suffix.merge(flags_agreeing).unwrap();
        assert_eq!(merged.bits(), 14);
    }

    #[test]
    fn config_defaults_depend_on_mode() {
        let classic = Config::new(test_args()).unwrap();
        assert!(classic.pattern.is_none());
        assert_eq!(classic.leading_zeroes_threshold, 3);
        assert_eq!(classic.total_zeroes_threshold, 5);

        let mut args = test_args();
        args.hook_flags = Some(0x2400);
        let pattern_mode = Config::new(args).unwrap();
        assert!(pattern_mode.pattern.is_some());
        assert_eq!(pattern_mode.leading_zeroes_threshold, 0);
        assert_eq!(pattern_mode.total_zeroes_threshold, 255);
    }

    #[test]
    fn config_combines_prefix_and_hook_flags() {
        let mut args = test_args();
        args.prefix = Some("0xc0ffee".into());
        args.hook_flags = Some(0x00c0);
        let config = Config::new(args).unwrap();
        let pattern = config.pattern.unwrap();
        assert_eq!(pattern.bits(), 24 + 14);
        assert_eq!(
            pattern.to_string(),
            "0xc0ffeexxxxxxxxxxxxxxxxxxxxxxxxxxxxxx?0c0"
        );

        let mut address = [0u8; 20];
        address[0] = 0xc0;
        address[1] = 0xff;
        address[2] = 0xee;
        address[19] = 0xc0;
        assert!(pattern.matches(&address));
    }

    #[test]
    fn kernel_src_embeds_pattern_conditions() {
        let mut args = test_args();
        args.hook_flags = Some(0x2400);
        let config = Config::new(args).unwrap();
        let src = mk_kernel_src(&config, KernelFlavor::OpenCl64);
        assert!(src.contains("#define PATTERN 1"));
        assert!(src.contains("#define hasPattern(d) ((((d)[18] & 63u) == 36u) && ((d)[19] == 0u))"));
        assert!(src.contains("#define LEADING_ZEROES 0"));
        assert!(src.contains("#define TOTAL_ZEROES 255"));

        let classic = Config::new(test_args()).unwrap();
        assert!(!mk_kernel_src(&classic, KernelFlavor::OpenCl64).contains("#define PATTERN 1"));
    }

    #[test]
    fn kernel_src_32bit_embeds_interleaved_constants() {
        let mut args = test_args();
        args.hook_flags = Some(0x2400);
        let config = Config::new(args).unwrap();
        let src = mk_kernel_src(&config, KernelFlavor::OpenCl32);

        // lane 0 = 0xff ++ factory[0..7]; the test factory is all 0x11
        let lane0 = u64::from_le_bytes([0xff, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11]);
        let (e, o) = bit_interleave(lane0);
        assert!(src.contains(&format!("#define A0_E {e}u")));
        assert!(src.contains(&format!("#define A0_O {o}u")));

        // padding lane 16 = 0x80 in its top byte
        let (e, o) = bit_interleave(0x8000000000000000);
        assert!(src.contains(&format!("#define A16_E {e}u")));
        assert!(src.contains(&format!("#define A16_O {o}u")));

        // all 23 iota constants, and the pattern check, are present
        let (e, o) = bit_interleave(KECCAK_RC[22]);
        assert!(src.contains(&format!("#define RC22_E {e}u")));
        assert!(src.contains(&format!("#define RC22_O {o}u")));
        assert!(src.contains("#define hasPattern"));
        assert!(src.contains("bit-interleaved"));
    }

    #[test]
    fn cli_parses_hook_flags_and_patterns() {
        use clap::Parser as _;
        let args = Args::try_parse_from([
            "create2crunch",
            "0x0000000000ffe8b47b3e2130213b802212439497",
            "0x0000000000000000000000000000000000000000",
            "0x2222222222222222222222222222222222222222222222222222222222222222",
            "--hook-flags",
            "0x3fff",
            "--prefix",
            "c0ffee",
        ])
        .unwrap();
        assert_eq!(args.hook_flags, Some(0x3fff));
        assert_eq!(args.gpu_device, 255);
        assert!(Args::try_parse_from([
            "create2crunch",
            "0x0000000000ffe8b47b3e2130213b802212439497",
            "0x0000000000000000000000000000000000000000",
            "0x2222222222222222222222222222222222222222222222222222222222222222",
            "--hook-flags",
            "0x4000",
        ])
        .is_err());
    }
}
