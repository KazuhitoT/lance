// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::path::{Path, PathBuf};
use std::sync::Arc;

use lance_core::{Error, Result};
use sentencepiece::SentencePieceProcessor;
use snafu::location;
use tantivy::tokenizer::{Token, TokenStream, Tokenizer};

pub const SENTENCEPIECE_MODEL_FILE: &str = "sentencepiece.model";

pub trait SentencePieceTokenizerBuilder: Sized {
    fn load(p: &Path) -> Result<Self> {
        if !p.is_dir() {
            return Err(Error::io(
                format!("{} is not a valid directory", p.display()),
                snafu::location!(),
            ));
        }
        let model_path = p.join(SENTENCEPIECE_MODEL_FILE);
        Self::new(model_path.as_path())
    }

    fn new(model_path: &Path) -> Result<Self>;

    fn build(&self) -> Result<tantivy::tokenizer::TextAnalyzerBuilder>;
}

pub struct SentencePieceBuilder {
    model_path: PathBuf,
}

#[derive(Clone)]
pub struct SentencePieceTokenizer {
    processor: Arc<SentencePieceProcessor>,
}

impl SentencePieceTokenizer {
    pub fn new(processor: SentencePieceProcessor) -> Self {
        Self {
            processor: Arc::new(processor),
        }
    }
}

impl Tokenizer for SentencePieceTokenizer {
    type TokenStream<'a> = SentencePieceTokenStream<'a>;

    fn token_stream<'a>(&mut self, text: &'a str) -> Self::TokenStream<'a> {
        SentencePieceTokenStream::new(text, Arc::clone(&self.processor))
    }
}

pub struct SentencePieceTokenStream<'a> {
    text: &'a str,
    processor: Arc<SentencePieceProcessor>,
    tokens: Vec<Token>,
    current_index: usize,
}

impl<'a> SentencePieceTokenStream<'a> {
    fn new(text: &'a str, processor: Arc<SentencePieceProcessor>) -> Self {
        let mut stream = Self {
            text,
            processor,
            tokens: Vec::new(),
            current_index: 0,
        };
        stream.tokenize();
        stream
    }

    fn tokenize(&mut self) {
        match self.processor.encode(self.text) {
            Ok(pieces) => {
                let mut char_offset = 0;
                for piece in pieces {
                    let token_text = piece.piece.clone();
                    let token_len = token_text.chars().count();

                    let mut token = Token::default();
                    token.text.clear();
                    token.text.push_str(&token_text);
                    token.offset_from = char_offset;
                    token.offset_to = char_offset + token_len;
                    token.position = self.tokens.len();

                    self.tokens.push(token);
                    char_offset += token_len;
                }
            }
            Err(e) => {
                log::error!("Failed to tokenize text: {}", e);
            }
        }
    }
}

impl<'a> TokenStream for SentencePieceTokenStream<'a> {
    fn advance(&mut self) -> bool {
        if self.current_index < self.tokens.len() {
            self.current_index += 1;
            true
        } else {
            false
        }
    }

    fn token(&self) -> &Token {
        &self.tokens[self.current_index.saturating_sub(1)]
    }

    fn token_mut(&mut self) -> &mut Token {
        let idx = self.current_index.saturating_sub(1);
        &mut self.tokens[idx]
    }
}

impl SentencePieceTokenizerBuilder for SentencePieceBuilder {
    fn new(model_path: &Path) -> Result<Self> {
        Ok(Self {
            model_path: model_path.to_path_buf(),
        })
    }

    fn build(&self) -> Result<tantivy::tokenizer::TextAnalyzerBuilder> {
        if !self.model_path.exists() {
            return Err(Error::io(
                format!(
                    "SentencePiece model file not found at '{}'",
                    self.model_path.display()
                ),
                location!(),
            ));
        }

        let processor = SentencePieceProcessor::open(&self.model_path).map_err(|e| {
            Error::io(
                format!(
                    "Failed to load SentencePiece model at {}: {}",
                    self.model_path.display(),
                    e
                ),
                location!(),
            )
        })?;

        let tokenizer = SentencePieceTokenizer::new(processor);
        Ok(tantivy::tokenizer::TextAnalyzer::builder(tokenizer).dynamic())
    }
}
