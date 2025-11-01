use clap::{Parser, Subcommand, ValueEnum};
use litert_lm::{LitManager, LiteRtMcpService, Result};

#[derive(Parser)]
#[command(name = "litert-lm")]
#[command(about = "LiteRT-LM wrapper with MCP and OpenAI-compatible APIs")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Clone, ValueEnum)]
enum McpTransport {
    /// Standard input/output (for local MCP clients)
    Stdio,
    /// Server-Sent Events over HTTP
    Sse,
    /// Resumable HTTP transport
    Http,
}

#[derive(Subcommand)]
enum Commands {
    /// List locally downloaded models
    List {
        /// List all models available for download in model registry
        #[arg(long)]
        show_all: bool,
    },
    /// Download a model from registry or URL
    Pull {
        model: String,
        /// Alias to save the model as (only for URLs)
        #[arg(long)]
        alias: Option<String>,
        /// Hugging Face API token for authentication
        #[arg(long)]
        hf_token: Option<String>,
    },
    /// Remove a locally downloaded model
    Rm { model: String },
    /// Run a LiteRT-LM model and start an interactive session
    Run { model: String },
    /// Generate completion script
    Completion { shell: String },
    /// Start OpenAI-compatible API server
    Serve {
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },
    /// Start MCP (Model Context Protocol) server
    Mcp {
        /// Transport method: stdio, sse, or http
        #[arg(short, long, default_value = "stdio")]
        transport: McpTransport,
        /// Port for SSE/HTTP transports (ignored for stdio)
        #[arg(short, long, default_value = "3000")]
        port: u16,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    let manager = LitManager::new().await?;

    match cli.command {
        Commands::List { show_all } => manager.list(show_all).await?,
        Commands::Pull { model, alias, hf_token } => manager.pull(&model, alias.as_deref(), hf_token.as_deref()).await?,
        Commands::Rm { model } => manager.remove(&model).await?,
        Commands::Run { model } => manager.run_interactive(&model).await?,
        Commands::Completion { shell } => manager.generate_completion(&shell)?,
        Commands::Serve { port } => manager.serve(port).await?,
        Commands::Mcp { transport, port } => {
            run_mcp_server(manager, transport, port).await?
        }
    }

    Ok(())
}

async fn run_mcp_server(
    manager: LitManager,
    transport: McpTransport,
    port: u16,
) -> Result<()> {
    use rmcp::ServiceExt;

    let service = LiteRtMcpService::new(manager).await?;

    match transport {
        McpTransport::Stdio => {
            tracing::info!("Starting MCP server with stdio transport");
            let (stdin, stdout) = (tokio::io::stdin(), tokio::io::stdout());
            service.serve((stdin, stdout)).await?;
        }
        McpTransport::Sse => {
            tracing::info!("Starting MCP server with SSE transport on port {}", port);

            // Create SSE server config
            let ct = tokio_util::sync::CancellationToken::new();
            let config = rmcp::transport::sse_server::SseServerConfig {
                bind: format!("0.0.0.0:{}", port).parse()?,
                sse_path: "/sse".to_string(),
                post_path: "/message".to_string(),
                ct: ct.clone(),
                sse_keep_alive: Some(std::time::Duration::from_secs(30)),
            };

            // Start SSE server
            let sse_server = rmcp::transport::sse_server::SseServer::serve_with_config(config).await?;

            // Serve with the service
            let _ct = sse_server.with_service_directly(move || service.clone());

            // Keep running
            tokio::signal::ctrl_c().await?;
        }
        McpTransport::Http => {
            // Note: Streamable HTTP transport requires session management and is more complex.
            // The SSE transport provides full HTTP-based MCP access with simpler setup.
            // For a full stateful HTTP implementation, you would need:
            // - A session manager (Arc<SessionManager>)
            // - StreamableHttpServerConfig
            // - A service factory function
            // Then wrap with hyper_util::service::TowerToHyperService for hyper 1.0 compatibility

            tracing::warn!("Stateful HTTP transport requires additional session management setup");
            tracing::info!("Use --transport sse for full HTTP-based MCP server support");
            tracing::info!("Falling back to stdio transport");

            let (stdin, stdout) = (tokio::io::stdin(), tokio::io::stdout());
            service.serve((stdin, stdout)).await?;
        }
    }

    Ok(())
}
