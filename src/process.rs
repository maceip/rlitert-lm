use anyhow::{Context, Result};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::Stream;

// Command sent to the process's internal loop
enum ProcessCommand {
    Run {
        prompt: String,
        // Send tokens back on this channel
        response_tx: mpsc::Sender<Result<String>>,
    },
}

pub struct LitProcess {
    // Kept to send commands *to* the process
    command_tx: mpsc::Sender<ProcessCommand>,
    // Kept for cleanup/shutdown, but not directly accessed in normal flow
    #[allow(dead_code)]
    child_handle: tokio::task::JoinHandle<()>,
}

impl std::fmt::Debug for LitProcess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LitProcess")
            .field("command_tx", &"<mpsc::Sender>")
            .field("child_handle", &"<JoinHandle>")
            .finish()
    }
}

impl LitProcess {
    pub async fn spawn(binary_path: PathBuf, model: String) -> Result<Self> {
        // Try GPU first, fall back to CPU if it fails
        match Self::spawn_with_backend(binary_path.clone(), model.clone(), "gpu").await {
            Ok(process) => Ok(process),
            Err(e) => {
                tracing::warn!("GPU backend failed: {}. Trying CPU backend...", e);
                Self::spawn_with_backend(binary_path, model, "cpu").await
            }
        }
    }

    async fn spawn_with_backend(binary_path: PathBuf, model: String, backend: &str) -> Result<Self> {
        tracing::info!("Attempting to spawn lit process with backend={}", backend);

        let mut child = Command::new(&binary_path)
            .arg("run")
            .arg(&model)
            .arg("--backend")
            .arg(backend)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("Failed to spawn lit process with backend={}", backend))?;

        let mut stdin = child.stdin.take().context("Failed to get stdin")?;
        let stdout = child.stdout.take().context("Failed to get stdout")?;
        let mut stderr = child.stderr.take().context("Failed to get stderr")?;

        let (command_tx, mut command_rx) = mpsc::channel::<ProcessCommand>(32);

        // Spawn a task to log stderr
        tokio::spawn(async move {
            use tokio::io::AsyncReadExt;
            let mut buf = [0u8; 1024];
            while let Ok(n) = stderr.read(&mut buf).await {
                if n == 0 {
                    break;
                }
                let msg = String::from_utf8_lossy(&buf[..n]);
                tracing::debug!("lit stderr: {}", msg.trim());
            }
        });

        // Spawn the long-running task that owns the process
        let child_handle = tokio::spawn(async move {
            use tokio::io::AsyncReadExt;

            let mut stdout = stdout;
            let mut buffer = Vec::new();
            let mut temp_buf = [0u8; 1024];
            let mut pending_commands = Vec::new();

            // Wait for model to load - look for the prompt marker ">>>"
            tracing::info!("Waiting for model to load...");
            let init_timeout = tokio::time::Duration::from_secs(120); // 2 minute timeout
            let init_result = tokio::time::timeout(init_timeout, async {
                loop {
                    tokio::select! {
                        // Check for incoming commands while initializing - buffer them
                        cmd = command_rx.recv() => {
                            if let Some(cmd) = cmd {
                                tracing::debug!("Buffering command during initialization");
                                pending_commands.push(cmd);
                            }
                        }
                        // Read from stdout
                        result = stdout.read(&mut temp_buf) => {
                            match result {
                                Ok(0) => {
                                    tracing::error!("Process stdout closed before model loaded");
                                    return Err(anyhow::anyhow!("Process died during initialization"));
                                }
                                Ok(n) => {
                                    buffer.extend_from_slice(&temp_buf[..n]);
                                    let text = String::from_utf8_lossy(&buffer);

                                    // Check for error messages
                                    if text.contains("Error") || text.contains("error") || text.contains("failed") {
                                        tracing::error!("Initialization error: {}", text);
                                        return Err(anyhow::anyhow!("Process initialization failed: {}", text.trim()));
                                    }

                                    // Check if model is loaded
                                    if text.contains("Model '") && text.contains("' loaded.") {
                                        tracing::info!("Model loaded successfully");
                                    }

                                    // Wait for the initial prompt marker
                                    if text.contains(">>>") {
                                        tracing::info!("Process ready to accept prompts");
                                        buffer.clear();
                                        return Ok(());
                                    }
                                }
                                Err(e) => {
                                    tracing::error!("Error reading process output during init: {}", e);
                                    return Err(e.into());
                                }
                            }
                        }
                    }
                }
            }).await;

            match init_result {
                Ok(Ok(())) => {
                    tracing::info!("Model initialization complete, processing {} buffered commands", pending_commands.len());
                }
                Ok(Err(e)) => {
                    tracing::error!("Initialization failed: {}", e);
                    // Drain buffered commands with error
                    for cmd in pending_commands {
                        let ProcessCommand::Run { response_tx, .. } = cmd;
                        let _ = response_tx.send(Err(anyhow::anyhow!("Process initialization failed: {}", e))).await;
                    }
                    let _ = child.kill().await;
                    return;
                }
                Err(_) => {
                    tracing::error!("Initialization timed out after 2 minutes");
                    for cmd in pending_commands {
                        let ProcessCommand::Run { response_tx, .. } = cmd;
                        let _ = response_tx.send(Err(anyhow::anyhow!("Process initialization timed out"))).await;
                    }
                    let _ = child.kill().await;
                    return;
                }
            }

            // Process any buffered commands first
            for cmd in pending_commands {
                Self::handle_command(cmd, &mut stdin, &mut stdout, &mut buffer, &mut temp_buf).await;
            }

            // Now handle commands
            while let Some(cmd) = command_rx.recv().await {
                Self::handle_command(cmd, &mut stdin, &mut stdout, &mut buffer, &mut temp_buf).await;
            }

            // Cleanup: kill child process when command loop exits
            let _ = child.kill().await;
        });

        Ok(Self {
            command_tx,
            child_handle,
        })
    }

    async fn handle_command(
        cmd: ProcessCommand,
        stdin: &mut tokio::process::ChildStdin,
        stdout: &mut tokio::process::ChildStdout,
        buffer: &mut Vec<u8>,
        temp_buf: &mut [u8; 1024],
    ) {
        use tokio::io::AsyncReadExt;

        match cmd {
            ProcessCommand::Run { prompt, response_tx } => {
                // 1. Write prompt to the process's stdin
                if let Err(e) = stdin.write_all(prompt.as_bytes()).await {
                    let _ = response_tx.send(Err(e.into())).await;
                    return;
                }
                if let Err(e) = stdin.write_all(b"\n").await {
                    let _ = response_tx.send(Err(e.into())).await;
                    return;
                }
                if let Err(e) = stdin.flush().await {
                    let _ = response_tx.send(Err(e.into())).await;
                    return;
                }

                // 2. Read character-by-character and stream tokens
                buffer.clear();
                let mut last_chunk = String::new();

                loop {
                    match stdout.read(temp_buf).await {
                        Ok(0) => {
                            // EOF - process died
                            let _ = response_tx.send(Err(anyhow::anyhow!("Process stdout closed"))).await;
                            break;
                        }
                        Ok(n) => {
                            buffer.extend_from_slice(&temp_buf[..n]);
                            let text = String::from_utf8_lossy(buffer).to_string();

                            // Check if we've reached the end marker ">>>"
                            if text.ends_with(">>>") || text.contains("\n>>>") {
                                // Send the final chunk (without the >>>)
                                let final_text = text.trim_end_matches(">>>").trim_end_matches('\n');
                                if final_text.len() > last_chunk.len() {
                                    let new_content = &final_text[last_chunk.len()..];
                                    if !new_content.is_empty() {
                                        if response_tx.send(Ok(new_content.to_string())).await.is_err() {
                                            break;
                                        }
                                    }
                                }
                                buffer.clear();
                                break;
                            }

                            // Send incremental updates
                            if text.len() > last_chunk.len() {
                                let new_content = &text[last_chunk.len()..];
                                if response_tx.send(Ok(new_content.to_string())).await.is_err() {
                                    // Client disconnected
                                    buffer.clear();
                                    break;
                                }
                                last_chunk = text;
                            }
                        }
                        Err(e) => {
                            let _ = response_tx.send(Err(e.into())).await;
                            break;
                        }
                    }
                }
                // When done, `response_tx` is dropped, closing the stream
            }
        }
    }

    // New streaming method
    pub async fn send_prompt_stream(
        &self,
        prompt: &str,
    ) -> Result<impl Stream<Item = Result<String>>> {
        // 1. Create a new, unique channel for *this* request's response
        let (response_tx, response_rx) = mpsc::channel(100); // Token buffer

        // 2. Create the command
        let cmd = ProcessCommand::Run {
            prompt: prompt.to_string(),
            response_tx,
        };

        // 3. Send the command to the process loop
        self.command_tx.send(cmd).await.map_err(|e| {
            // Process loop died
            anyhow::anyhow!("Failed to send command to process: {}", e)
        })?;

        // 4. Return the receiver wrapped in a stream
        Ok(ReceiverStream::new(response_rx))
    }

    // Keep the old non-streaming method for backward compatibility
    pub async fn send_prompt(&self, prompt: &str) -> Result<String> {
        use futures::StreamExt;

        let mut stream = self.send_prompt_stream(prompt).await?;
        let mut response = String::new();

        while let Some(result) = stream.next().await {
            let line = result?;
            response.push_str(&line);
            response.push('\n');
        }

        Ok(response)
    }

    #[allow(dead_code)]
    pub async fn shutdown(self) -> Result<()> {
        // Drop command_tx to signal shutdown
        drop(self.command_tx);
        // Wait for child task to finish
        self.child_handle.await?;
        Ok(())
    }
}

/// Manages a pool of isolated LitProcess instances
#[derive(Debug)]
pub struct ProcessPool {
    binary_path: PathBuf,
    model: String,
    processes: Vec<Arc<LitProcess>>,
}

impl ProcessPool {
    pub fn new(binary_path: PathBuf, model: String, pool_size: usize) -> Self {
        Self {
            binary_path,
            model,
            processes: Vec::with_capacity(pool_size),
        }
    }

    pub async fn initialize(&mut self) -> Result<()> {
        let pool_size = self.processes.capacity();
        for _ in 0..pool_size {
            let process = LitProcess::spawn(self.binary_path.clone(), self.model.clone()).await?;
            self.processes.push(Arc::new(process));
        }
        Ok(())
    }

    pub async fn get_process(&self) -> Result<Arc<LitProcess>> {
        // Simple round-robin selection
        // In a real implementation, you might want to track which processes are busy
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);

        if self.processes.is_empty() {
            anyhow::bail!("Process pool not initialized")
        }

        let idx = COUNTER.fetch_add(1, Ordering::Relaxed) % self.processes.len();
        Ok(self.processes[idx].clone())
    }

    pub async fn send_prompt(&self, prompt: &str) -> Result<String> {
        let process = self.get_process().await?;
        process.send_prompt(prompt).await
    }
}
