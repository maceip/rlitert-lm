use anyhow::{Context, Result};
use std::env;
use std::fs;
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;

const VERSION: &str = "v0.7.0";
const BASE_URL: &str = "https://github.com/google-ai-edge/LiteRT-LM/releases/download";

#[derive(Debug, Clone)]
pub struct BinaryManager {
    cache_dir: PathBuf,
}

impl BinaryManager {
    pub fn new() -> Result<Self> {
        let cache_dir = dirs::cache_dir()
            .context("Failed to get cache directory")?
            .join("litert-lm");

        tracing::debug!(cache_dir = %cache_dir.display(), "Setting up binary manager");
        fs::create_dir_all(&cache_dir)?;
        tracing::trace!(cache_dir = %cache_dir.display(), "Cache directory ready");

        Ok(Self { cache_dir })
    }

    pub async fn ensure_binary(&self) -> Result<PathBuf> {
        let binary_path = self.get_binary_path();

        if binary_path.exists() {
            tracing::debug!(path = %binary_path.display(), "Binary already exists");
            return Ok(binary_path);
        }

        tracing::info!(path = %binary_path.display(), "Binary not found, downloading...");
        self.download_binary(&binary_path).await?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            tracing::debug!("Setting executable permissions");
            let mut perms = fs::metadata(&binary_path)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&binary_path, perms)?;
        }

        tracing::info!(path = %binary_path.display(), "Binary ready");
        Ok(binary_path)
    }

    fn get_binary_path(&self) -> PathBuf {
        let filename = self.get_binary_filename();
        self.cache_dir.join(filename)
    }

    fn get_binary_filename(&self) -> &'static str {
        match (env::consts::OS, env::consts::ARCH) {
            ("linux", "aarch64") => "lit.linux_arm64",
            ("linux", "x86_64") => "lit.linux_x86_64",
            ("macos", "aarch64") => "lit.macos_arm64",
            ("windows", "x86_64") => "lit.windows_x86_64.exe",
            _ => panic!("Unsupported platform: {}/{}", env::consts::OS, env::consts::ARCH),
        }
    }

    async fn download_binary(&self, dest: &PathBuf) -> Result<()> {
        let filename = self.get_binary_filename();
        let url = format!("{}/{}/{}", BASE_URL, VERSION, filename);

        tracing::info!(url = %url, "Downloading binary");

        let response = reqwest::get(&url)
            .await
            .context("Failed to download binary")?;

        if !response.status().is_success() {
            tracing::error!(
                url = %url,
                status = %response.status(),
                "Download request failed"
            );
            anyhow::bail!("Failed to download binary: HTTP {}", response.status());
        }

        tracing::debug!("Download response received, reading bytes");
        let bytes = response.bytes().await?;
        tracing::debug!(size_bytes = bytes.len(), "Binary downloaded, writing to disk");

        let mut file = tokio::fs::File::create(dest).await?;
        file.write_all(&bytes).await?;
        file.flush().await?;

        tracing::info!(
            path = %dest.display(),
            size_bytes = bytes.len(),
            "Binary downloaded successfully"
        );
        Ok(())
    }
}
