//use std::fs::File;
//use std::io::Read;
//use std::fs;
//use std::error::Error;
//use std::fmt;
//use std::io;
use std::path::{Path,PathBuf};

extern crate clap;
use clap::{Arg, App, SubCommand};

extern crate rust_bert;
use rust_bert::pipelines::token_classification::{TokenClassificationOption,ModelType};

extern crate serde;
extern crate serde_yaml;
#[macro_use]
extern crate serde_derive;

#[derive(Serialize, Deserialize)]
struct Configuration {
    models: Vec<ModelConfiguration>
}

#[derive(Serialize, Deserialize)]
struct ModelConfiguration {

    ///FoLiA Annotation type
    annotation_type: String,

    ///FoLiA Set Definition
    folia_set: String;

    ///Model name (for local)
    model_name: String

    ///Model Type (bert, roberta, distilbert)
    model_type: String,

    ///Model url (model.ot as downloadable from Huggingface)
    model_url: String,

    ///Config url (config.json as downloadable from Huggingface)
    config_url: String,

    ///Vocab url (vocab.txt/vocab.json as downloadable from Huggingface)
    vocab_url: String,

    ///Merges url (merges.txt as downloadable from Huggingface, for Roberta)
    merges_url: String,

}



/*
impl DeepFrogResources {
    pub const POS: (&'static str, &'static str) = ("bert-ner/model.ot", "https://s3.amazonaws.com/models.huggingface.co/bert/dbmdz/bert-large-cased-finetuned-conll03-english/rust_model.ot");
}
*/

fn main() {
    println!("Hello, world!");
    let matches = App::new("DeepFrog")
        .version("0.1")
        .author("Maarten van Gompel (proycon) <proycon@anaproy.nl>")
        .about("An NLP tool")
            //snippet hints --> addargb,addargs,addargi,addargf,addargpos
            .arg(Arg::with_name("model")
                .long("--model")
                .short("-m")
                .help("The model to use")
                .takes_value(true)
                .value_name("DIR")
                .required(true))
            .arg(Arg::with_name("file")
                .help("Input file")
                .takes_value(true)
                .index(1)
                .required(true))
        .get_matches();

    let modeldir = PathBuf::from(matches.value_of("model").expect("No model supplied"));

    if !PathBuf::from(&modeldir).is_dir() {
        eprintln!("ERROR: Model {} not found (or not a directory)", modeldir.to_str().expect("path to string"));
        std::process::exit(2);
    }

    let config = NERConfig::new(
        model_resource:
    }

}
