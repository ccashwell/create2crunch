#![warn(unused_crate_dependencies, unreachable_pub)]
#![deny(unused_must_use, rust_2018_idioms)]

use alloy_primitives::{hex, Address, FixedBytes};
use byteorder::{BigEndian, ByteOrder, LittleEndian};
use clap::Parser;
use console::Term;
use fs4::FileExt;
use keccak_asm::{Digest, Keccak256};
use ocl::{Buffer, Context, Device, MemFlags, Platform, ProQue, Program, Queue};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use separator::Separatable;
use std::error::Error;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};
use terminal_size::{terminal_size, Height};

mod reward;
pub use reward::Reward;

// workset size (tweak this!)
const WORK_SIZE: u32 = 0x4000000; // max. 0x15400000 to abs. max 0xffffffff

const WORK_FACTOR: u128 = (WORK_SIZE as u128) / 1_000_000;
const CONTROL_CHARACTER: u8 = 0xff;
const MAX_INCREMENTER: u64 = 0xffffffffffff;

static KERNEL_SRC: &str = include_str!("./kernels/keccak256.cl");

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
pub struct Config {
    pub factory_address: [u8; 20],
    pub calling_address: [u8; 20],
    pub init_code_hash: [u8; 32],
    pub gpu_device: u8,
    pub leading_zeroes_threshold: u8,
    pub total_zeroes_threshold: u8,
    pub pattern: Option<Pattern>,
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
        })
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
pub fn cpu(config: Config) -> Result<(), Box<dyn Error>> {
    // (create if necessary) and open a file where found salts will be written
    let file = output_file();

    // create object for computing rewards (relative rarity) for a given address
    let rewards = Reward::new();

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

        // iterate over a 6-byte nonce and compute each address
        (0..MAX_INCREMENTER)
            .into_par_iter() // parallelization
            .for_each(|salt| {
                let salt = salt.to_le_bytes();
                let salt_incremented_segment = &salt[..6];

                // splice the nonce into the message and hash it
                let mut message = template;
                message[47..53].copy_from_slice(salt_incremented_segment);
                let res = Keccak256::digest(message);

                // get the address that results from the hash
                let address = <&Address>::try_from(&res[12..]).unwrap();

                // count total and leading zero bytes
                let mut total = 0;
                let mut leading = 21;
                for (i, &b) in address.iter().enumerate() {
                    if b == 0 {
                        total += 1;
                    } else if leading == 21 {
                        // set leading on finding non-zero byte
                        leading = i;
                    }
                }

                let key = leading * 20 + total;
                let reward_amount = match &config.pattern {
                    Some(pattern) => {
                        // pattern mode: require the pattern plus any
                        // explicitly-provided zero-byte thresholds
                        if !pattern.matches(&res[12..]) {
                            return;
                        }
                        if (leading.min(20) as u8) < config.leading_zeroes_threshold {
                            return;
                        }
                        if config.total_zeroes_threshold <= 20
                            && (total as u8) < config.total_zeroes_threshold
                        {
                            return;
                        }
                        rewards.get(&key)
                    }
                    None => {
                        // only proceed if there are at least three zero bytes
                        if total < 3 {
                            return;
                        }

                        // only proceed if an efficient address has been found
                        let reward_amount = rewards.get(&key);
                        if reward_amount.is_none() {
                            return;
                        }
                        reward_amount
                    }
                };

                // get the full salt used to create the address
                let full_salt = format!(
                    "0x{}{}",
                    hex::encode(&template[21..47]),
                    hex::encode(salt_incremented_segment)
                );

                // display the salt and the address.
                let output = format!(
                    "{full_salt} => {address} => {}",
                    reward_amount.unwrap_or("0")
                );
                println!("{output}");

                // create a lock on the file before writing
                file.lock_exclusive().expect("Couldn't lock file.");

                // write the result to file
                writeln!(&file, "{output}")
                    .expect("Couldn't write to `efficient_addresses.txt` file.");

                // release the file lock
                file.unlock().expect("Couldn't unlock file.")
            });
    }
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
pub fn gpu(config: Config) -> ocl::Result<()> {
    println!(
        "Setting up experimental OpenCL miner using device {}...",
        config.gpu_device
    );

    // (create if necessary) and open a file where found salts will be written
    let file = output_file();

    // create object for computing rewards (relative rarity) for a given address
    let rewards = Reward::new();

    // track how many addresses have been found and information about them
    let mut found: u64 = 0;
    let mut found_list: Vec<String> = vec![];

    // set up a controller for terminal output
    let term = Term::stdout();

    // set up a platform to use
    let platform = Platform::new(ocl::core::default_platform()?);

    // set up the device to use
    let device = Device::by_idx_wrap(platform, config.gpu_device as usize)?;

    // set up the context to use
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    // set up the program to use
    let program = Program::builder()
        .devices(device)
        .src(mk_kernel_src(&config))
        .build(&context)?;

    // set up the queue to use
    let queue = Queue::new(&context, device, None)?;

    // set up the "proqueue" (or amalgamation of various elements) to use
    let ocl_pq = ProQue::new(context, queue, program, Some(WORK_SIZE));

    // create a random number generator
    let mut rng = thread_rng();

    // determine the start time
    let start_time: f64 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    // set up variables for tracking performance
    let mut rate: f64 = 0.0;
    let mut cumulative_nonce: u64 = 0;

    // the previous timestamp of printing to the terminal
    let mut previous_time: f64 = 0.0;

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

        // reset nonce & create a buffer to view it in little-endian
        // for more uniformly distributed nonces, we shall initialize it to a random value
        let mut nonce: [u32; 1] = rng.gen();
        let mut view_buf = [0; 8];

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
            let current_time = now.as_secs() as f64;

            // we don't want to print too fast
            let print_output = current_time - previous_time > 0.99;
            previous_time = current_time;

            // clear the terminal screen
            if print_output {
                term.clear_screen()?;

                // get the total runtime and parse into hours : minutes : seconds
                let total_runtime = current_time - start_time;
                let total_runtime_hrs = total_runtime as u64 / 3600;
                let total_runtime_mins = (total_runtime as u64 - total_runtime_hrs * 3600) / 60;
                let total_runtime_secs = total_runtime
                    - (total_runtime_hrs * 3600) as f64
                    - (total_runtime_mins * 60) as f64;

                // determine the number of attempts being made per second
                let work_rate: u128 = WORK_FACTOR * cumulative_nonce as u128;
                if total_runtime > 0.0 {
                    rate = 1.0 / total_runtime;
                }

                // fill the buffer for viewing the properly-formatted nonce
                LittleEndian::write_u64(&mut view_buf, (nonce[0] as u64) << 32);

                // calculate the terminal height, defaulting to a height of ten rows
                let height = terminal_size().map(|(_w, Height(h))| h).unwrap_or(10);

                // display information about the total runtime and work size
                term.write_line(&format!(
                    "total runtime: {}:{:02}:{:02} ({} cycles)\t\t\t\
                     work size per cycle: {}",
                    total_runtime_hrs,
                    total_runtime_mins,
                    total_runtime_secs,
                    cumulative_nonce,
                    WORK_SIZE.separated_string(),
                ))?;

                // display information about the attempt rate and found solutions
                term.write_line(&format!(
                    "rate: {:.2} million attempts per second\t\t\t\
                     total found this run: {}",
                    work_rate as f64 * rate,
                    found
                ))?;

                // display information about the current search criteria
                match &config.pattern {
                    Some(pattern) => term.write_line(&format!(
                        "current search space: {}xxxxxxxx{:08x}\t\t\
                         target pattern: {} (2^{})",
                        hex::encode(salt),
                        BigEndian::read_u64(&view_buf),
                        pattern,
                        pattern.bits(),
                    ))?,
                    None => term.write_line(&format!(
                        "current search space: {}xxxxxxxx{:08x}\t\t\
                         threshold: {} leading or {} total zeroes",
                        hex::encode(salt),
                        BigEndian::read_u64(&view_buf),
                        config.leading_zeroes_threshold,
                        config.total_zeroes_threshold
                    ))?,
                }

                // display recently found solutions based on terminal height
                let rows = if height < 5 { 1 } else { height as usize - 4 };
                let last_rows: Vec<String> = found_list.iter().cloned().rev().take(rows).collect();
                let ordered: Vec<String> = last_rows.iter().cloned().rev().collect();
                let recently_found = &ordered.join("\n");
                term.write_line(recently_found)?;
            }

            // increment the cumulative nonce (does not reset after a match)
            cumulative_nonce += 1;

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
            if solution == 0 {
                continue;
            }

            let solution = solution.to_le_bytes();

            let mut solution_message = [0; 85];
            solution_message[0] = CONTROL_CHARACTER;
            solution_message[1..21].copy_from_slice(&config.factory_address);
            solution_message[21..41].copy_from_slice(&config.calling_address);
            solution_message[41..45].copy_from_slice(&salt[..]);
            solution_message[45..53].copy_from_slice(&solution);
            solution_message[53..].copy_from_slice(&config.init_code_hash);

            // hash the payload and get the result
            let res = Keccak256::digest(solution_message);

            // get the address that results from the hash
            let address = <&Address>::try_from(&res[12..]).unwrap();

            // re-verify the pattern on the host before recording the result
            if let Some(pattern) = &config.pattern {
                if !pattern.matches(&res[12..]) {
                    continue;
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
            found_list.push(show.to_string());

            file.lock_exclusive().expect("Couldn't lock file.");

            writeln!(&file, "{output}").expect("Couldn't write to `efficient_addresses.txt` file.");

            file.unlock().expect("Couldn't unlock file.");
            found += 1;
        }
    }
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

/// Creates the OpenCL kernel source code by populating the template with the
/// values from the Config object.
fn mk_kernel_src(config: &Config) -> String {
    let mut src = String::with_capacity(2048 + KERNEL_SRC.len());

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

    src.push_str(KERNEL_SRC);

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
        let src = mk_kernel_src(&config);
        assert!(src.contains("#define PATTERN 1"));
        assert!(src.contains("#define hasPattern(d) ((((d)[18] & 63u) == 36u) && ((d)[19] == 0u))"));
        assert!(src.contains("#define LEADING_ZEROES 0"));
        assert!(src.contains("#define TOTAL_ZEROES 255"));

        let classic = Config::new(test_args()).unwrap();
        assert!(!mk_kernel_src(&classic).contains("#define PATTERN 1"));
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
