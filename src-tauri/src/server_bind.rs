use tauri::AppHandle;

pub const LOOPBACK_HOST: &str = "127.0.0.1";

/// The Security+ profile is intentionally loopback-only. Persisted settings,
/// environment variables, and legacy LAN flags cannot widen this boundary.
pub fn configured_bind_host(_app: &AppHandle) -> String {
    LOOPBACK_HOST.to_string()
}

pub fn bind_addr(host: &str, port: u16) -> String {
    format!("{host}:{port}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_loopback_is_compiled_into_the_profile() {
        assert_eq!(LOOPBACK_HOST, "127.0.0.1");
        assert_eq!(bind_addr(LOOPBACK_HOST, 19828), "127.0.0.1:19828");
    }
}
