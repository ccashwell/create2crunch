//! Native Metal compute backend for Apple GPUs.
//!
//! macOS's OpenCL implementation is deprecated and runs as a shim over
//! Metal; this backend drives the same bit-interleaved keccak kernel (as
//! MSL) through Metal directly. Buffers use the unified-memory shared
//! storage mode, so solutions are read back without any copies, and each
//! dispatch simply blocks on completion instead of the OpenCL path's
//! sleep-and-poll loop.

use crate::{
    mk_kernel_src, output_file, record_solution, Config, FoundList, KernelFlavor, Reward,
    StatusDisplay, WORK_SIZE,
};
use alloy_primitives::FixedBytes;
use metal::{Device, MTLResourceOptions, MTLSize};
use objc::rc::autoreleasepool;
use rand::{thread_rng, Rng};
use std::error::Error;

/// Search for salts on an Apple GPU via Metal. Mirrors the OpenCL `gpu`
/// loop: an outer loop draws a random 4-byte salt segment, an inner loop
/// dispatches `WORK_SIZE` hashes per command buffer while stepping the
/// host nonce word, and any found solution is re-verified on the host and
/// recorded.
pub(crate) fn gpu(config: Config, found_list: Option<FoundList>) -> Result<(), Box<dyn Error>> {
    let devices = Device::all();
    let device = devices
        .get(config.gpu_device as usize)
        .ok_or_else(|| format!("no Metal device at index {}", config.gpu_device))?;
    println!(
        "Setting up Metal miner using device {} ({})...",
        config.gpu_device,
        device.name()
    );
    println!("Using the bit-interleaved 32-bit keccak kernel...");

    // (create if necessary) and open a file where found salts will be written
    let file = output_file();

    // create object for computing rewards (relative rarity) for a given address
    let rewards = Reward::new();

    // track found addresses (shared with a concurrent CPU miner when --cpu)
    let found_list = found_list.unwrap_or_default();

    let library = device
        .new_library_with_source(
            &mk_kernel_src(&config, KernelFlavor::Metal),
            &metal::CompileOptions::new(),
        )
        .map_err(|e| format!("Metal kernel compilation failed: {e}"))?;
    let function = library
        .get_function("hashMessage", None)
        .map_err(|e| format!("Metal kernel function lookup failed: {e}"))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("Metal pipeline creation failed: {e}"))?;
    let queue = device.new_command_queue();

    // unified memory: the host writes inputs and reads solutions directly
    let shared = MTLResourceOptions::StorageModeShared;
    let message_buffer = device.new_buffer(4, shared);
    let nonce_buffer = device.new_buffer(4, shared);
    let solutions_buffer = device.new_buffer(8, shared);

    let threads_per_grid = MTLSize::new(WORK_SIZE as u64, 1, 1);
    let threads_per_threadgroup =
        MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1);

    // create a random number generator
    let mut rng = thread_rng();

    // set up the terminal status display and performance tracking
    let mut display = StatusDisplay::new();
    let mut cumulative_nonce: u64 = 0;

    // begin searching for addresses
    loop {
        // construct the 4-byte message to hash, leaving last 8 of salt empty
        let salt = FixedBytes::<4>::random();
        unsafe {
            std::ptr::copy_nonoverlapping(salt.as_ptr(), message_buffer.contents() as *mut u8, 4);
            *(solutions_buffer.contents() as *mut u64) = 0;
        }

        // reset the nonce; for more uniformly distributed nonces, we shall
        // initialize it to a random value
        let mut nonce: u32 = rng.gen();

        // repeatedly dispatch the kernel to search for new addresses
        let solution = loop {
            unsafe {
                *(nonce_buffer.contents() as *mut u32) = nonce;
            }

            // command buffers are autoreleased objects; drain each iteration
            autoreleasepool(|| {
                let command_buffer = queue.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_buffer(0, Some(&message_buffer), 0);
                encoder.set_buffer(1, Some(&nonce_buffer), 0);
                encoder.set_buffer(2, Some(&solutions_buffer), 0);
                encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
                encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            });

            // increment the cumulative nonce (does not reset after a match)
            cumulative_nonce += 1;

            // repaint the status screen (at most once per second)
            display.maybe_print(&config, &found_list, cumulative_nonce, &salt[..], nonce)?;

            let solution = unsafe { *(solutions_buffer.contents() as *const u64) };
            if solution != 0 {
                break solution;
            }

            // if no solution has yet been found, increment the nonce
            nonce = nonce.wrapping_add(1);
        };

        record_solution(&config, &rewards, &file, &found_list, &salt[..], solution);
    }
}
