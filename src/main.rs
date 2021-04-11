//use std::fs::File;
//use std::io::Read;
//use std::fs;
//use std::fmt;
//use std::io;
use std::path::PathBuf;
use std::error::Error;
use std::str;

extern crate clap;
use clap::{Arg, App};

extern crate deepfrog;
use deepfrog::{DeepFrog,consolidate_layers};


fn main() -> Result<(), Box<dyn Error + 'static>> {
    let matches = App::new("DeepFrog")
        .version("0.3.0")
        .author("Maarten van Gompel (proycon) <proycon@anaproy.nl>")
        .about("An NLP tool")
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
            .arg(Arg::with_name("json-low")
                .long("--json-low")
                .short("-j")
                .help("Output low-level JSON directly after prediction")
                .required(false))
            .arg(Arg::with_name("json")
                .long("--json")
                .short("-J")
                .help("Output higher-level JSON, includes consolidation of layers")
                .required(false))
            .arg(Arg::with_name("xml")
                .long("--xml")
                .short("-x")
                .help("Output FoLiA XML")
                .required(false))
        .get_matches();

    if !matches.is_present("json-low") && !matches.is_present("json") && !matches.is_present("xml") {
        eprintln!("ERROR: Please specify an output option: --xml, --json-low, json");
        std::process::exit(2);
    }

    let configfile = PathBuf::from(matches.value_of("config").expect("No configuration file supplied"));


    let files: Vec<_> = matches.values_of("files").unwrap().collect();
    //checking input files exist
    let mut all_input_found = true;
    for (_, filename) in files.iter().enumerate() {
        if !PathBuf::from(filename).exists() {
            eprintln!("ERROR: Input {} not found", filename);
            all_input_found = false;
        }
    }
    if !all_input_found {
        std::process::exit(2);
    }


    let mut deepfrog = DeepFrog::from_config(&configfile)?;
    deepfrog.load_models()?;



    for (i, filename) in files.iter().enumerate() {
        eprintln!("Processing file {} of {}: {} ...", i+1, files.len(), filename);
        let (mut output, input) = deepfrog.process_text(&filename, true)?;
        eprintln!("\t{} input lines, {} output lines ...", input.len(), output.len());
        if matches.is_present("json-low") {
            eprintln!("\tOutputting low-level JSON");
            DeepFrog::print_json_low(&output, &input);
        } else {
            let offsets_to_tokens = consolidate_layers(&output);
            deepfrog.translate_labels(&input, &mut output, &offsets_to_tokens);
            if matches.is_present("json") {
                eprintln!("\tOutputting high-level JSON");
                deepfrog.print_json_high(&offsets_to_tokens, &output, &input);
            } else {
                eprintln!("\tOutputting FoLiA XML");
                let filename = PathBuf::from(filename);
                let id = if let Some(filestem) = filename.file_stem() {
                    filestem.to_str().expect("to str")
                } else {
                    "undefined"
                };
                let doc = deepfrog.to_folia(id, &offsets_to_tokens, &output, &input)?;
                println!("{}",str::from_utf8(&doc.xml(0,4).expect("serialising to XML")).expect("parsing utf-8"));
            }
        }
    }


    Ok(())
}






