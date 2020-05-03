//use std::fs::File;
//use std::io::Read;
//use std::fs;
//use std::fmt;
//use std::io;
use std::path::{Path,PathBuf};
use std::fs;
use std::error::Error;
use std::str::FromStr;
use std::collections::{HashMap,HashSet};

extern crate clap;
use clap::{Arg, App, SubCommand};

extern crate rust_bert;
use rust_bert::pipelines::token_classification::{TokenClassificationConfig, TokenClassificationOption,ConfigOption,TokenizerOption,ModelType, TokenClassificationModel};

extern crate serde;
extern crate serde_yaml;
#[macro_use]
extern crate serde_derive;


#[derive(Serialize, Deserialize)]
struct Configuration {
    language: String,
    models: Vec<ModelSpecification>
}

#[derive(Serialize, Deserialize)]
struct ModelSpecification {

    ///FoLiA Annotation type
    annotation_type: String,

    ///FoLiA Set Definition
    folia_set: String,

    ///Model name (for local)
    model_name: String,

    ///Model Type (bert, roberta, distilbert)
    model_type: ModelType,

    ///Model local name (including directory part), no absolute path, automatically derived if not specified
    #[serde(default)]
    model_local: String,

    ///Model url (model.ot as downloadable from Huggingface)
    model_remote: String,

    ///Config local name (including directory part), no absolute path, automatically derived if not specified
    #[serde(default)]
    config_local: String,

    ///Config url (config.json as downloadable from Huggingface)
    config_remote: String,

    ///Config local name (including directory part), no absolute path, automatically derived if not specified
    #[serde(default)]
    vocab_local: String,

    ///Vocab url (vocab.txt/vocab.json as downloadable from Huggingface)
    vocab_remote: String,

    ///Merges local name  (xxx/merges.txt)
    #[serde(default)]
    merges_local: Option<String>,

    ///Merges url (merges.txt as downloadable from Huggingface, for Roberta)
    #[serde(default)]
    merges_remote: Option<String>,

}


struct Model {
    model: TokenClassificationOption,
    config: ConfigOption,
    tokenizer: TokenizerOption,
}

fn main() -> Result<(), Box<dyn Error + 'static>> {
    let matches = App::new("DeepFrog")
        .version("0.1")
        .author("Maarten van Gompel (proycon) <proycon@anaproy.nl>")
        .about("An NLP tool")
            //snippet hints --> addargb,addargs,addargi,addargf,addargpos
            .arg(Arg::with_name("config")
                .long("--config")
                .short("-c")
                .help("The DeepFrog configuration to use")
                .takes_value(true)
                .value_name("FILE")
                .required(true))
            .arg(Arg::with_name("files")
                .help("Input files")
                .takes_value(true)
                .multiple(true)
                .required(true))
        .get_matches();

    let configfile = PathBuf::from(matches.value_of("config").expect("No configuration file supplied"));

    if !PathBuf::from(&configfile).exists() {
        eprintln!("ERROR: Configuration {} not found", configfile.to_str().expect("path to string"));
        std::process::exit(2);
    }

    let configdata = fs::read_to_string(&configfile)?;
    let config: Configuration = serde_yaml::from_str(configdata.as_str()).expect("Invalid yaml in configuration file");
    eprintln!("Loaded configuration: {}", &configfile.to_str().expect("path to string conversion"));

    let mut token_classification_models: Vec<TokenClassificationModel> = Vec::new();

    for (i, modelspec) in config.models.iter().enumerate() {
        eprintln!("    Loading model {} of {}: {}:{}  ...", i+1, config.models.len(), modelspec.annotation_type, modelspec.model_name);
        let merges: Option<(&str,&str)> = if let (Some(merges_local), Some(merges_remote)) = (modelspec.merges_local.as_ref(), modelspec.merges_remote.as_ref()) {
            Some( (merges_local.as_str(), merges_remote.as_str()) )
        } else {
            None
        };
        //Load the actual configuration
        let modelconfig = TokenClassificationConfig::new(modelspec.model_type, (modelspec.model_local.as_str(), modelspec.model_remote.as_str()), (modelspec.config_local.as_str(), modelspec.config_remote.as_str()), (modelspec.vocab_local.as_str(), modelspec.vocab_remote.as_str()), merges);

        //Load the model
        if let Ok(model) = TokenClassificationModel::new(modelconfig) {
            token_classification_models.push(model);
        } else {
            eprintln!("        Error loading model!");
        }
    }

    let files: Vec<_> = matches.values_of("files").unwrap().collect();

    for (i, filename) in files.iter().enumerate() {
        eprintln!("Processing file {} of {}: {} ...", i+1, files.len(), filename);
    }


    Ok(())
}
