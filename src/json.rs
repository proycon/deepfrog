use crate::{ModelOutput,OffsetToTokens,get_text_by_char_offset};

use crate::DeepFrog;

impl DeepFrog {

    ///Low-level JSON output, directly as outputted by the underlying model
    pub fn print_json_low(output: &Vec<ModelOutput>, input: &Vec<String>) {
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
    pub fn print_json_high(&self, offsets_to_tokens: &Vec<OffsetToTokens>, output: &Vec<ModelOutput>, input: &Vec<String>) {
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
                let model_name = &self.config.models.get(*model_index).expect("getting model").model_name;
                let token = &output[*model_index].labeled_tokens[*token_index];
                println!("        {{ \"model_name\": \"{}\", \"label\": \"{}\", \"confidence\": {} }},",  model_name, token.label, token.score);
            }
            println!("    ]");
            println!("  }},");
        }
        println!("}}");
    }
}


