use thiserror::Error;

#[derive(Error, Debug)]
pub enum Ms3Error {
    #[error("Gateway error: {0}")]
    Gateway(String),

    #[error("Personality not found: {0}")]
    PersonalityNotFound(String),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Persistence error: {0}")]
    Persistence(String),

    #[error("Ethics violation: {0}")]
    EthicsViolation(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

pub type Ms3Result<T> = Result<T, Ms3Error>;
