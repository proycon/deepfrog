use folia::ReadElement;
use crate::{OffsetToTokens,ModelOutput,ModelSpecification,get_text_by_char_offset};

use std::collections::HashMap;

use crate::DeepFrog;
use crate::error::DeepFrogError;

struct SpanBuffer {
    begin: usize, //begin token nr
    sentence: usize, //sentence nr
    class: String,
    ids: Vec<String>,
    confidence: f64,
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

impl DeepFrog {
    ///Instantiate processors for provenance keeping
    pub fn folia_processor(&self) -> folia::Processor {
        let mut processor = folia::Processor::new(format!("deepfrog"))
                                     .with_version(format!("{}",env!("CARGO_PKG_VERSION")))
                                     .with_src(format!("https://proycon.github.com/deepfrog"))
                                     .with_format(format!("text/html"))
                                     .autofill();
        for subprocessor in self.subprocessors.iter() {
            processor = processor.with_new_subprocessor(subprocessor.clone());
        }
        processor
    }


    ///Consolidate the output of multiple models into one structure
    pub fn to_folia(&self, id: &str, offsets_to_tokens: &Vec<OffsetToTokens>, output: &Vec<ModelOutput>, input: &Vec<String>) -> Result<folia::Document,DeepFrogError> {
        let processor = self.folia_processor();
        let mut doc = folia::Document::new(id, folia::DocumentProperties::default().with_processor(processor) )?;
        //create sentences
        let root: folia::ElementKey = 0; //root element always has key 0
        let mut sentence_nr = 0;
        let mut word_nr = 0;
        let mut sentence_index = 9999;
        let mut sentence_key: folia::ElementKey = 0;

        doc.declare(folia::AnnotationType::SENTENCE, &None, &None, &None)?;
        doc.declare(folia::AnnotationType::TOKEN, &None, &None, &None)?;

        let mut spanbuffers_permodel: HashMap<usize,Vec<SpanBuffer>> = HashMap::new();

        //add the tokens
        for offset in offsets_to_tokens.into_iter() {
            if sentence_index != offset.sentence {
                if !spanbuffers_permodel.is_empty() {
                    flush_spanbuffers_to_folia(&mut doc, &spanbuffers_permodel, &self.config.models);
                    spanbuffers_permodel.clear();
                }
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

                let modelspec = &self.config.models.get(*model_index).expect("getting model");

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
                    if let None = spanbuffers_permodel.get(model_index) {
                        //make sure we have a spanbuffer entry for this model
                        spanbuffers_permodel.insert(*model_index, Vec::new());
                    }
                    if let Some(spanbuffers) = spanbuffers_permodel.get_mut(model_index) {
                        let l = spanbuffers.len();
                        let mut addnew = true;
                        if l > 0 {
                            //get the last item from the buffer, see if the current token is part of it
                            if let Some(spanbuffer) = spanbuffers.get_mut(l -1) {
                                if !forcenew && spanbuffer.class == class && spanbuffer.sentence == sentence_nr && spanbuffer.begin + spanbuffer.ids.len()  == word_nr {
                                    //increase the coverage of the existing buffer to include the current word
                                    spanbuffer.ids.push(doc.get_element(word_key).expect("element").id().expect("id").to_string());
                                    spanbuffer.confidence *= token.score;
                                    addnew = false;
                                }
                            }
                        }
                        if addnew {
                            spanbuffers.push(SpanBuffer {
                                begin: word_nr,
                                sentence: sentence_nr,
                                class: class,
                                ids: vec!(doc.get_element(word_key).expect("element").id().expect("id").to_string()),
                                confidence: token.score,
                            });
                        }
                    }
                } else {
                    eprintln!("WARNING: Can't handle element type {} yet", element_type);
                }

            }
        }

        //flush all span buffers
        flush_spanbuffers_to_folia(&mut doc, &spanbuffers_permodel, &self.config.models);

        Ok(doc)
    }
}

fn flush_spanbuffers_to_folia(doc: &mut folia::Document, spanbuffers_permodel: &HashMap<usize,Vec<SpanBuffer>>, models: &Vec<ModelSpecification>) {
    for (model_index, spanbuffers) in spanbuffers_permodel.iter() {
         for spanbuffer in spanbuffers {
            let modelspec = &models.get(*model_index).expect("getting model");
            let element_type = modelspec.annotation_type.elementtype();
            spanbuffer.to_folia(doc, element_type, modelspec.folia_set.to_owned());
        }
    }
}

