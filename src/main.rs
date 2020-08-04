//use std::fs::File;
//use std::io::Read;
//use std::fs;
//use std::fmt;
//use std::io;
use std::path::{Path,PathBuf};
use std::fs;
use std::fs::File;
use std::io::{Write,Read,BufReader,BufRead};
use std::error::Error;
use std::str;
use std::collections::{HashMap,HashSet};

extern crate clap;
use clap::{Arg, App, SubCommand};

extern crate rust_bert;
use rust_bert::pipelines::common::{ModelType,TokenizerOption,ConfigOption};
use rust_bert::pipelines::token_classification::{TokenClassificationConfig,TokenClassificationModel, Token, LabelAggregationOption};
use rust_bert::resources::{Resource,RemoteResource};

extern crate serde;
extern crate serde_yaml;
extern crate serde_json;
#[macro_use]
extern crate serde_derive;

extern crate folia;
use folia::ReadElement;


#[derive(Serialize, Deserialize)]
struct Configuration {
    language: String,
    models: Vec<ModelSpecification>
}

#[derive(Serialize, Deserialize)]
struct ModelSpecification {

    ///FoLiA Annotation type
    annotation_type: folia::AnnotationType,

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

    ///Ignored label, you can set it to something like "O" or "Outside" for NER tasks, depending
    ///on how your tagset denotes non-entities.
    #[serde(default)]
    ignore_label: String,

    ///Indicates if this is a lower-cased model, in which case all input will be automatically lower-cased
    #[serde(default)]
    lowercase: bool,

    ///Does this model adhere to the BIO-scheme?
    #[serde(default)]
    bio: bool,

    ///Delimiter used in the BIO-scheme (example: in a tag like B-per the delimiter is a hyphen)
    #[serde(default)]
    bio_delimiter: String,

}

struct ModelOutput<'a> {
    model_name: &'a str,
    labeled_tokens: Vec<Token>,
}

struct SpanBuffer {
    begin: usize, //begin token nr
    sentence: usize, //sentence nr
    class: String,
    ids: Vec<String>,
    confidence: f64,
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

    if !PathBuf::from(&configfile).exists() {
        eprintln!("ERROR: Configuration {} not found", configfile.to_str().expect("path to string"));
        std::process::exit(2);
    }

    let files: Vec<_> = matches.values_of("files").unwrap().collect();
    //checking input files exist
    let mut all_input_found = true;
    for (i, filename) in files.iter().enumerate() {
        if !PathBuf::from(filename).exists() {
            eprintln!("ERROR: Input {} not found", filename);
            all_input_found = false;
        }
    }
    if !all_input_found {
        std::process::exit(2);
    }



    let configdata = fs::read_to_string(&configfile)?;
    let config: Configuration = serde_yaml::from_str(configdata.as_str()).expect("Invalid yaml in configuration file");
    eprintln!("Loaded configuration: {}", &configfile.to_str().expect("path to string conversion"));

    let mut token_classification_models: Vec<TokenClassificationModel> = Vec::new();

    for (i, modelspec) in config.models.iter().enumerate() {
        eprintln!("    Loading model {} of {}: {}:{}  ...", i+1, config.models.len(), modelspec.annotation_type, modelspec.model_name);

        let model_resource: Resource =  Resource::Remote(RemoteResource::from_pretrained( (modelspec.model_local.as_str(), modelspec.model_remote.as_str()) ));
        let config_resource: Resource =  Resource::Remote(RemoteResource::from_pretrained( (modelspec.config_local.as_str(), modelspec.config_remote.as_str()) ));
        let vocab_resource: Resource =  Resource::Remote(RemoteResource::from_pretrained( (modelspec.vocab_local.as_str(), modelspec.vocab_remote.as_str()) ));
        let merges_resource: Option<Resource> = if let (Some(merges_local), Some(merges_remote)) = (modelspec.merges_local.as_ref(), modelspec.merges_remote.as_ref()) {
            Some(Resource::Remote(RemoteResource::from_pretrained( (merges_local.as_str(), merges_remote.as_str()) )))
        } else {
            None
        };
        //Load the actual configuration
        let modelconfig = TokenClassificationConfig::new(modelspec.model_type, model_resource, config_resource, vocab_resource, merges_resource, modelspec.lowercase, LabelAggregationOption::Mode);

        //Load the model
        if let Ok(model) = TokenClassificationModel::new(modelconfig) {
            token_classification_models.push(model);
        } else {
            eprintln!("ERROR: Failed to load model!");
            std::process::exit(3);
        }
    }


    for (i, filename) in files.iter().enumerate() {
        eprintln!("Processing file {} of {}: {} ...", i+1, files.len(), filename);
        let result = process_text(&filename, &config, &token_classification_models, true);
        let (output, input) = result.expect("unwrapping");
        if matches.is_present("json-low") {
            print_json_low(&output, &input);
        } else {
            let offsets_to_tokens = consolidate_layers(&output);
            if matches.is_present("json") {
                print_json_high(&offsets_to_tokens, &output, &input, &config.models);
            } else {
                let filename = PathBuf::from(filename);
                let id = if let Some(filestem) = filename.file_stem() {
                    filestem.to_str().expect("to str")
                } else {
                    "undefined"
                };
                let mut doc = folia::Document::new(id, folia::DocumentProperties::default() )?;
                doc = to_folia(doc, &offsets_to_tokens, &output, &input, &config.models);
                println!("{}",str::from_utf8(&doc.xml(0,4).expect("serialising to XML")).expect("parsing utf-8"));
            }
        }
    }


    Ok(())
}

///Low-level JSON output, directly as outputted by the underlying model
fn print_json_low(output: &Vec<ModelOutput>, input: &Vec<String>) {
    println!("{{");
    println!("\"input\": {}", serde_json::to_string(&input).expect("json"));
    println!("\"output_layers\": [");
    for output_layer in output {
        println!("  {{ \"model_name\": \"{}\", \"tokens\": [", output_layer.model_name);
        for token in output_layer.labeled_tokens.iter() {
            print!("    {}", serde_json::to_string(&token).expect("json"));
            println!(",")
        }
        println!("  ] }},");
    }
    println!("]}}");
}

///High-level JSON output, output after consolidation of annotation layers
fn print_json_high(offsets_to_tokens: &Vec<OffsetToTokens>, output: &Vec<ModelOutput>, input: &Vec<String>, models: &Vec<ModelSpecification>) {
    println!("{{");
    println!("\"input\": {}", serde_json::to_string(&input).expect("json"));
    println!("\"output_tokens\": [");
    for offset in offsets_to_tokens {
        let sentence_text = input.get(offset.sentence).expect("sentence not found in input");
        let token_text: &str = get_text_by_char_offset(sentence_text, offset.begin, offset.end).expect("unable to get token text");
        println!("  {{ \"text\": \"{}\",", token_text);
        println!("    \"offset\": {{ \"sentence\": {}, \"begin\": {}, \"end\": {} }},", offset.sentence, offset.begin, offset.end);
        println!("    \"annotations\": [");
        for (model_index, token_index) in offset.model_token_indices.iter() {
            let model_name = &models.get(*model_index).expect("getting model").model_name;
            let token = &output[*model_index].labeled_tokens[*token_index];
            println!("        {{ \"model_name\": \"{}\", \"label\": \"{}\", \"confidence\": {} }},",  model_name, token.label, token.score);
        }
        println!("    ]");
        println!("  }},");
    }
    println!("}}");
}

fn get_text_by_char_offset(s: &str, begin: u32, end: u32) -> Option<&str> {
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



fn process_text<'a>(filename: &str, config: &'a Configuration, models: &Vec<TokenClassificationModel>, return_input: bool) -> Result<(Vec<ModelOutput<'a>>, Vec<String>), Box<dyn Error + 'static>> {
    let f = File::open(filename)?;
    let f_buffer = BufReader::new(f);
    let lines: Vec<String> = f_buffer.lines().map(|s| s.unwrap()).collect();
    let lines_ref: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
    let mut output: Vec<ModelOutput> = Vec::new();
    for (model, modelspec) in models.iter().zip(config.models.iter()) {
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


struct OffsetToTokens {
    sentence: usize,
    begin: u32,
    end: u32,
    model_token_indices: Vec<(usize,usize)>, //model_index + token_index
}


///Consolidate annotations in multiple layers, creating a reverse index
///from offsets to models and labelled tokens
fn consolidate_layers(output: &Vec<ModelOutput>) -> Vec<OffsetToTokens> {
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


///Consolidate the output of multiple models into one structure
fn to_folia(mut doc: folia::Document, offsets_to_tokens: &Vec<OffsetToTokens>, output: &Vec<ModelOutput>, input: &Vec<String>, models: &Vec<ModelSpecification>) -> folia::Document {
    //create sentences
    let root: folia::ElementKey = 0; //root element always has key 0
    let mut sentence_nr = 0;
    let mut word_nr = 0;
    let mut sentence_index = 9999;
    let mut sentence_key: folia::ElementKey = 0;

    doc.declare(folia::AnnotationType::SENTENCE, &None, &None, &None);
    doc.declare(folia::AnnotationType::TOKEN, &None, &None, &None);

    let mut spanbuffer_permodel: HashMap<usize,SpanBuffer> = HashMap::new();

    //add the tokens
    for offset in offsets_to_tokens.into_iter() {
        if sentence_index != offset.sentence {
            //new sentence
            sentence_nr += 1;
            word_nr = 0; //reset
            sentence_key = doc.annotate(root,
                            folia::ElementData::new(folia::ElementType::Sentence).
                            with_attrib(folia::Attribute::Id(format!("{}.s.{}", doc.id(), sentence_nr).to_string()))
                            ).expect("Adding sentence");

            sentence_index = offset.sentence;
        }

        assert_ne!(sentence_key,0);

        let sentence_text = input.get(offset.sentence).expect("sentence not found in input");
        let token_text: &str = get_text_by_char_offset(sentence_text, offset.begin, offset.end).expect("unable to get token text");

        //add words/tokens
        word_nr += 1;
        let word_key = doc.annotate(sentence_key,
                            folia::ElementData::new(folia::ElementType::Word)
                            .with_attrib(folia::Attribute::Id(format!("{}.s.{}.w.{}", doc.id(), sentence_nr, word_nr).to_string()))

                            ).expect("Adding word");

        doc.annotate(word_key,
                     folia::ElementData::new(folia::ElementType::TextContent)
                     .with(folia::DataType::Text(token_text.to_owned()))
                     ).expect("Adding word");

        for (model_index, token_index) in offset.model_token_indices.iter() {
            let token = &output[*model_index].labeled_tokens[*token_index];

            let modelspec = &models.get(*model_index).expect("getting model");

            let element_type = modelspec.annotation_type.elementtype();

            if folia::ElementGroup::Inline.contains(element_type) {
                doc.annotate(word_key,
                             folia::ElementData::new(element_type)
                             .with_attrib(folia::Attribute::Set(modelspec.folia_set.to_owned())) //can be more efficient by using keys
                             .with_attrib(folia::Attribute::Class(token.label.clone()))
                            ).expect("Adding inline annotation");

            } else if folia::ElementGroup::Span.contains(element_type) {
                let (class, forcenew) = if modelspec.bio {
                    if token.label.starts_with(format!("B{}", modelspec.bio_delimiter).as_str()) {
                        (token.label[1 + modelspec.bio_delimiter.len()..].to_owned(), true)
                    } else if token.label.starts_with(format!("I{}", modelspec.bio_delimiter).as_str()) {
                        (token.label[1 + modelspec.bio_delimiter.len()..].to_owned(), false)
                    } else if token.label == modelspec.ignore_label {
                        continue;
                    } else {
                        (token.label.clone(), false)
                    }
                } else {
                    (token.label.clone(), false)
                };
                if let Some(spanbuffer) = spanbuffer_permodel.get_mut(model_index) {
                    if forcenew || spanbuffer.class != class || spanbuffer.sentence != sentence_nr || spanbuffer.begin + spanbuffer.ids.len() + 1  != word_nr {
                        //flush the existing buffer
                        spanbuffer.to_folia(&mut doc, element_type, modelspec.folia_set.to_owned());
                        //and start a new one
                        *spanbuffer = SpanBuffer {
                            begin: word_nr,
                            sentence: sentence_nr,
                            class: class,
                            ids: vec!(doc.get_element(word_key).expect("element").id().expect("id").to_string()),
                            confidence: token.score,
                        }
                    } else {
                        //increase the coverage of the existing buffer to include the current word
                        spanbuffer.ids.push(doc.get_element(word_key).expect("element").id().expect("id").to_string());
                        spanbuffer.confidence *= token.score;
                    }
                } else {
                    spanbuffer_permodel.insert(*model_index,
                        SpanBuffer {
                            begin: word_nr,
                            sentence: sentence_nr,
                            class: class,
                            ids: vec!(doc.get_element(word_key).expect("element").id().expect("id").to_string()),
                            confidence: token.score,
                        }
                    );
                }
            } else {
                eprintln!("WARNING: Can't handle element type {} yet", element_type);
            }

        }
    }

    //flush remaining span buffers
    for (model_index, spanbuffer) in spanbuffer_permodel.iter() {
        let modelspec = &models.get(*model_index).expect("getting model");
        let element_type = modelspec.annotation_type.elementtype();
        spanbuffer.to_folia(&mut doc, element_type, modelspec.folia_set.to_owned());
    }

    doc
}

impl SpanBuffer {
    fn to_folia(&self, doc: &mut folia::Document, element_type: folia::ElementType, set: String) {
        doc.annotate_span(
                     folia::ElementData::new(element_type)
                     .with_attrib(folia::Attribute::Set(set))
                     .with_attrib(folia::Attribute::Class(self.class.clone()))
                     .with_span(&self.ids.iter().map(|s| s.as_str()).collect::<Vec<&str>>())
                    ).expect("Adding span annotation");
    }
}
