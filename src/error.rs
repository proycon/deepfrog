use thiserror::Error;
use rust_bert::RustBertError;
use std::fmt;
use folia::error::FoliaError;

#[derive(Error,Debug)]
pub enum DeepFrogError {
    IoError(String),
    FoliaError(String),
    RustBertError(String),
    OtherError(String)
}

impl From<std::io::Error> for DeepFrogError {
    fn from(error: std::io::Error) -> Self {
        DeepFrogError::IoError(error.to_string())
    }
}

impl From<RustBertError> for DeepFrogError {
    fn from(error: RustBertError) -> Self {
        DeepFrogError::RustBertError(error.to_string())
    }
}

impl From<FoliaError> for DeepFrogError {
    fn from(error: FoliaError) -> Self {
        DeepFrogError::FoliaError(error.to_string())
    }
}

impl From<String> for DeepFrogError {
    fn from(error: String) -> Self {
        DeepFrogError::OtherError(error)
    }
}

impl fmt::Display for DeepFrogError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::IoError(s) | Self::FoliaError(s) | Self::RustBertError(s) | Self::OtherError(s) => write!(f,"DeepFrog Error: {}",s)
        }
    }
}
