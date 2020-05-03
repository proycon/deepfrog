//use std::fs::File;
//use std::io::Read;
//use std::fs;
//use std::fmt;
//use std::io;
use std::path::{Path,PathBuf};
use std::fs;

extern crate clap;
use clap::{Arg, App, SubCommand};

extern crate rust_bert;
use rust_bert::pipelines::token_classification::{TokenClassificationOption,ModelType};

extern crate serde;
extern crate serde_yaml;
#[macro_use]
extern crate serde_derive;

use std::error::Error;

#[derive(Serialize, Deserialize)]
struct Configuration {
    language: String,
    models: Vec<ModelConfiguration>
}

#[derive(Serialize, Deserialize)]
struct ModelConfiguration {

    ///FoLiA Annotation type
    annotation_type: String,

    ///FoLiA Set Definition
    folia_set: String,

    ///Model name (for local)
    model_name: String,

    ///Model Type (bert, roberta, distilbert)
    model_type: String,

    ///Model url (model.ot as downloadable from Huggingface)
    model_url: String,

    ///Config url (config.json as downloadable from Huggingface)
    config_url: String,

    ///Vocab url (vocab.txt/vocab.json as downloadable from Huggingface)
    vocab_url: String,

    ///Merges url (merges.txt as downloadable from Huggingface, for Roberta)
    #[serde(default)]
    merges_url: Option<String>,

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
            .arg(Arg::with_name("file")
                .help("Input file")
                .takes_value(true)
                .index(1)
                .required(true))
        .get_matches();

    let configfile = PathBuf::from(matches.value_of("config").expect("No configuration file supplied"));

    if !PathBuf::from(&configfile).exists() {
        eprintln!("ERROR: Configuration {} not found", configfile.to_str().expect("path to string"));
        std::process::exit(2);
    }

    let configdata = fs::read_to_string(configfile)?;
    let config: Configuration = serde_yaml::from_str(configdata.as_str()).expect("Invalid yaml in configuration file");

    Ok(())
}
