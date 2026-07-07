use create2crunch::Config;
use std::process;

fn main() {
    let config = Config::parse_args().unwrap_or_else(|err| {
        eprintln!("Failed parsing arguments: {err}");
        process::exit(1);
    });

    if let Err(e) = create2crunch::run(config) {
        eprintln!("application error: {e}");
        process::exit(1);
    }
}
