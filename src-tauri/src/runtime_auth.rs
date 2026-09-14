use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde_json::json;

const RUNTIME_DIR_PREFIX: &str = "llm-wiki-sec-training-";
const RUNTIME_FILE_NAME: &str = "runtime.json";
const RUNTIME_DIRECTORY_ATTEMPTS: usize = 16;

struct RuntimeAuth {
    token: String,
    directory: PathBuf,
}

static RUNTIME_AUTH: OnceLock<RuntimeAuth> = OnceLock::new();

pub fn initialize() -> Result<(), String> {
    if RUNTIME_AUTH.get().is_some() {
        return Ok(());
    }

    let token = generate_token()?;
    let directory = create_runtime_directory()?;
    let runtime_file = directory.join(RUNTIME_FILE_NAME);
    if let Err(error) = write_runtime_file(&runtime_file, &token) {
        let _ = fs::remove_dir(&directory);
        return Err(error);
    }

    RUNTIME_AUTH
        .set(RuntimeAuth { token, directory })
        .map_err(|_| "Local runtime authentication was initialized twice".to_string())
}

pub fn token() -> Option<&'static str> {
    RUNTIME_AUTH.get().map(|runtime| runtime.token.as_str())
}

pub fn runtime_file() -> Option<PathBuf> {
    RUNTIME_AUTH
        .get()
        .map(|runtime| runtime.directory.join(RUNTIME_FILE_NAME))
}

pub fn cleanup() {
    let Some(runtime) = RUNTIME_AUTH.get() else {
        return;
    };
    if is_runtime_directory(&runtime.directory) {
        let _ = fs::remove_dir_all(&runtime.directory);
    }
}

fn create_runtime_directory() -> Result<PathBuf, String> {
    let parent = std::env::temp_dir();
    for _ in 0..RUNTIME_DIRECTORY_ATTEMPTS {
        let suffix = random_hex(16)?;
        let directory = parent.join(format!("{RUNTIME_DIR_PREFIX}{suffix}"));
        match fs::create_dir(&directory) {
            Ok(()) => {
                if let Err(error) = set_directory_permissions(&directory) {
                    let _ = fs::remove_dir(&directory);
                    return Err(error);
                }
                return Ok(directory);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("Failed to create local runtime directory: {error}")),
        }
    }
    Err("Failed to allocate a unique local runtime directory".to_string())
}

fn is_runtime_directory(path: &Path) -> bool {
    path.parent() == Some(std::env::temp_dir().as_path())
        && path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.starts_with(RUNTIME_DIR_PREFIX))
            .unwrap_or(false)
}

fn random_hex(byte_count: usize) -> Result<String, String> {
    let mut bytes = vec![0_u8; byte_count];
    getrandom::fill(&mut bytes)
        .map_err(|error| format!("OS randomness is unavailable: {error}"))?;
    let mut encoded = String::with_capacity(byte_count * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}")
            .map_err(|error| format!("Failed to encode runtime randomness: {error}"))?;
    }
    Ok(encoded)
}

fn generate_token() -> Result<String, String> {
    random_hex(32)
}

fn write_runtime_file(path: &Path, token: &str) -> Result<(), String> {
    let body = json!({
        "apiBase": "http://127.0.0.1:19828",
        "profile": "sec-training-local-v1",
        "token": token,
    })
    .to_string();

    let mut options = fs::OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .map_err(|err| format!("Failed to create runtime credential file: {err}"))?;
    file.write_all(body.as_bytes())
        .map_err(|err| format!("Failed to write runtime credential file: {err}"))?;
    file.sync_all()
        .map_err(|err| format!("Failed to sync runtime credential file: {err}"))
}

#[cfg(unix)]
fn set_directory_permissions(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .map_err(|err| format!("Failed to protect runtime directory: {err}"))
}

#[cfg(not(unix))]
fn set_directory_permissions(_path: &Path) -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_token_is_256_bits_of_hex() {
        let token = generate_token().unwrap();
        assert_eq!(token.len(), 64);
        assert!(token.chars().all(|character| character.is_ascii_hexdigit()));
    }

    #[test]
    fn runtime_directories_are_unique_and_owned_by_their_creator() {
        let first = create_runtime_directory().unwrap();
        let second = create_runtime_directory().unwrap();
        assert_ne!(first, second);
        assert!(is_runtime_directory(&first));
        assert!(is_runtime_directory(&second));
        fs::remove_dir(first).unwrap();
        fs::remove_dir(second).unwrap();
    }
}
