

struct EditRule<'a> {
    parts: Vec<EditInstruction<'a>>;
}

enum EditInstruction<'a> {
    Remove(&'a str),
    Add(&'a str),
    Keep(&'a str),
    KeepLength(usize)
}

///applies the edit rule (as generated by sesdiff) to convert a word from into its lemma
fn get_lemma(word: &str, editrule: &str) -> String {
       let parts: Vec<&str> = Vec::new();
       for c in
}

