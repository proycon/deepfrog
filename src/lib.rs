pub mod error;
pub mod foliaxml;
pub mod json;
pub mod lemma;

extern crate serde;
extern crate serde_yaml;
extern crate serde_json;
#[macro_use]
extern crate serde_derive;

extern crate folia;

extern crate rust_bert;
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::token_classification::{TokenClassificationConfig,TokenClassificationModel, Token, LabelAggregationOption};
use rust_bert::resources::{Resource,RemoteResource};

use std::fs::{File,read_to_string};
use std::io::{BufReader,BufRead};
use std::error::Error;
use std::path::{PathBuf};

use crate::error::DeepFrogError;

pub struct DeepFrog {
    pub config: Configuration,
    pub token_classification_models: Vec<TokenClassificationModel>,
    subprocessors: Vec<folia::Processor>
}


impl DeepFrog {
    pub fn from_config(configfile: &std::path::PathBuf) -> Result<Self,DeepFrogError> {
        if !PathBuf::from(&configfile).exists() {
            return Err(DeepFrogError::IoError(format!("Configuration {} not found", configfile.to_str().expect("path to string"))));
        }
        let configdata = read_to_string(&configfile)?;
        let config: Configuration = serde_yaml::from_str(configdata.as_str()).expect("Invalid yaml in configuration file");
        eprintln!("Loaded DeepFrog configuration: {}", &configfile.to_str().expect("path to string conversion"));
        Ok(DeepFrog {
            config: config,
            token_classification_models: Vec::new(),
            subprocessors: Vec::new(),
        })
    }

    pub fn load_models(&mut self) -> Result<(), DeepFrogError> {
        for (i, modelspec) in self.config.models.iter().enumerate() {
            eprintln!("    Loading model {} of {}: {}:{}  ...", i+1, self.config.models.len(), modelspec.annotation_type, modelspec.model_name);

            let model_resource: Resource =  Resource::Remote(RemoteResource::from_pretrained( (modelspec.model_local.as_str(), modelspec.model_remote.as_str()) ));
            let config_resource: Resource =  Resource::Remote(RemoteResource::from_pretrained( (modelspec.config_local.as_str(), modelspec.config_remote.as_str()) ));
            let vocab_resource: Resource =  Resource::Remote(RemoteResource::from_pretrained( (modelspec.vocab_local.as_str(), modelspec.vocab_remote.as_str()) ));
            let merges_resource: Option<Resource> = if let (Some(merges_local), Some(merges_remote)) = (modelspec.merges_local.as_ref(), modelspec.merges_remote.as_ref()) {
                Some(Resource::Remote(RemoteResource::from_pretrained( (merges_local.as_str(), merges_remote.as_str()) )))
            } else {
                None
            };
            //Load the actual configuration
            let modelconfig = TokenClassificationConfig::new(modelspec.model_type, model_resource, config_resource, vocab_resource, merges_resource, modelspec.lowercase, None, None, LabelAggregationOption::Mode);

            self.subprocessors.push(folia::Processor::new(format!("model-{}", modelspec.model_name))
                                .with_type(folia::ProcessorType::DataSource)
                                .with_src(if !modelspec.model_remote.is_empty() {
                                    modelspec.model_remote.clone()
                                } else {
                                    modelspec.model_local.clone()
                                }
                                )
                                .with_format(format!("application/libtorch"))
            );
            self.subprocessors.push(folia::Processor::new(format!("modelconfig-{}", modelspec.model_name))
                                .with_type(folia::ProcessorType::DataSource)
                                .with_src(if !modelspec.config_remote.is_empty() {
                                    modelspec.config_remote.clone()
                                } else {
                                    modelspec.config_local.clone()
                                }
                                )
                                .with_format(format!("application/json"))
            );
            self.subprocessors.push(folia::Processor::new(format!("modelvocab-{}", modelspec.model_name))
                                .with_type(folia::ProcessorType::DataSource)
                                .with_src(if !modelspec.vocab_remote.is_empty() {
                                    modelspec.vocab_remote.clone()
                                } else {
                                    modelspec.vocab_local.clone()
                                }
                                )
            );


            //Load the model
            if let Ok(model) = TokenClassificationModel::new(modelconfig) {
                self.token_classification_models.push(model);
            } else {
                return Err(DeepFrogError::OtherError("Failed to load model!".to_string()));
            }
        }
        Ok(())
    }

    pub fn process_text<'a>(&'a self, filename: &str, return_input: bool) -> Result<(Vec<ModelOutput<'a>>, Vec<String>), Box<dyn Error + 'static>> {
        let f = File::open(filename)?;
        let f_buffer = BufReader::new(f);
        let lines: Vec<String> = f_buffer.lines().map(|s| s.unwrap()).collect();
        let lines_ref: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let mut output: Vec<ModelOutput> = Vec::new();
        for (model, modelspec) in self.token_classification_models.iter().zip(self.config.models.iter()) {
            let labeled_tokens = model.predict(&lines_ref, true, false);
            output.push( ModelOutput {
                model_name: &modelspec.model_name,
                labeled_tokens: labeled_tokens
            });
        }
        let input: Vec<String> = if return_input {
            lines_ref.into_iter().map(|s| s.to_owned()).collect()
        } else {
            vec!()
        };
        Ok((output, input))
    }
}

#[derive(Serialize, Deserialize)]
pub struct Configuration {
    pub language: String,
    pub models: Vec<ModelSpecification>
}

#[derive(Serialize, Deserialize)]
pub struct ModelSpecification {

    ///FoLiA Annotation type
    pub annotation_type: folia::AnnotationType,

    ///FoLiA Set Definition
    pub folia_set: String,

    ///Model name (for local)
    pub model_name: String,

    ///Model Type (bert, roberta, distilbert)
    pub model_type: ModelType,

    ///Model local name (including directory part), no absolute path, automatically derived if not specified
    #[serde(default)]
    pub model_local: String,

    ///Model url (model.ot as downloadable from Huggingface)
    pub model_remote: String,

    ///Config local name (including directory part), no absolute path, automatically derived if not specified
    #[serde(default)]
    pub config_local: String,

    ///Config url (config.json as downloadable from Huggingface)
    pub config_remote: String,

    ///Config local name (including directory part), no absolute path, automatically derived if not specified
    #[serde(default)]
    pub vocab_local: String,

    ///Vocab url (vocab.txt/vocab.json as downloadable from Huggingface)
    pub vocab_remote: String,

    ///Merges local name  (xxx/merges.txt)
    #[serde(default)]
    pub merges_local: Option<String>,

    ///Merges url (merges.txt as downloadable from Huggingface, for Roberta)
    #[serde(default)]
    pub merges_remote: Option<String>,

    ///Ignored label, you can set it to something like "O" or "Outside" for NER tasks, depending
    ///on how your tagset denotes non-entities.
    #[serde(default)]
    pub ignore_label: String,

    ///Indicates if this is a lower-cased model, in which case all input will be automatically lower-cased
    #[serde(default)]
    pub lowercase: bool,

    ///Does this model adhere to the BIO-scheme?
    #[serde(default)]
    pub bio: bool,

    ///Delimiter used in the BIO-scheme (example: in a tag like B-per the delimiter is a hyphen)
    #[serde(default)]
    pub bio_delimiter: String,

}

pub struct ModelOutput<'a> {
    pub model_name: &'a str,
    pub labeled_tokens: Vec<Token>,
}


pub struct OffsetToTokens {
    pub sentence: usize,
    pub begin: u32,
    pub end: u32,
    pub model_token_indices: Vec<(usize,usize)>, //model_index + token_index
}


///Consolidate annotations in multiple layers, creating a reverse index
///from offsets to models and labelled tokens
pub fn consolidate_layers(output: &Vec<ModelOutput>) -> Vec<OffsetToTokens> {
    //create a vector of offsets
    if output.is_empty() {
        return vec!();
    }
    let mut offsets_to_tokens: Vec<OffsetToTokens> = Vec::with_capacity(output[0].labeled_tokens.len());
    for (i, output_layer) in output.iter().enumerate() {
        if i == 0 {
            //first tokenisation is the pivot for all
            for (j, token) in output_layer.labeled_tokens.iter().enumerate() {
                if let Some(offset) = token.offset {
                    offsets_to_tokens.push(OffsetToTokens {
                        sentence: token.sentence,
                        begin: offset.begin,
                        end: offset.end,
                        model_token_indices: vec!( (i,j) )
                    });
                }
            }
        } else {
            //consolidate with previous offsets
            for (j,token) in output_layer.labeled_tokens.iter().enumerate() {
                let mut consolidated = false;
                let mut begin: Option<usize> = None;
                let mut end: Option<usize> = None;
                if token.offset.is_some() {
                    for (k, offset) in offsets_to_tokens.iter_mut().enumerate() {
                        if token.sentence == offset.sentence {
                            if token.offset.unwrap().begin >= offset.begin && token.offset.unwrap().end <= offset.end {
                                offset.model_token_indices.push((i,j));
                                consolidated = true;
                                break;
                            } else if begin.is_none() && token.offset.unwrap().begin >= offset.begin && token.offset.unwrap().begin <= offset.end {
                                begin = Some(k);
                            } else if token.offset.unwrap().end >= offset.begin && token.offset.unwrap().end <= offset.end {
                                end = Some(k);
                            } else if offset.end > token.offset.unwrap().end  {
                                break;
                            }
                        }
                    }
                }
                if !consolidated {
                    //no exact or sub-match, let's try super-matches (this token matching multiple tokens)
                    if begin.is_some() && end.is_some() {
                        let begin = begin.unwrap();
                        let end = end.unwrap();
                        for k in begin..end {
                            if let Some(offset) = offsets_to_tokens.get_mut(k) {
                                offset.model_token_indices.push((i,j));
                            }
                        }
                    } else {
                        eprintln!("Token can not be consolidated with initial tokenisation {:?}:", token);
                    }
                }
            }
        }
    }
    offsets_to_tokens
}


///Returns a slice of
pub fn get_text_by_char_offset(s: &str, begin: u32, end: u32) -> Option<&str> {
    let mut bytebegin = 0;
    let mut byteend = 0;
    let mut charcount = 0;
    for (i, (byte, _)) in s.char_indices().enumerate() {
        charcount += 1;
        if i == begin as usize {
            bytebegin = byte;
        }
        if i == end as usize {
            byteend = byte;
            break;
        }
    }
    if byteend == 0 && end == charcount {
        byteend = s.len();
    }
    if bytebegin != byteend {
        Some(&s[bytebegin..byteend])
    } else {
        None
    }
}
