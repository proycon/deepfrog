#!/usr/bin/env nextflow

/*
vim: syntax=groovy
-*- mode: groovy;-*-
*/

log.info "----------------------------------"
log.info "DeepFrog Tagger/NER trainer"
log.info "----------------------------------"

def env = System.getenv()

params.virtualenv =  env.containsKey('VIRTUAL_ENV') ? env['VIRTUAL_ENV'] : ""

params.workers = Runtime.runtime.availableProcessors()
params.per_gpu_train_batch_size = 32
params.num_train_epochs = 3
params.save_steps = 20000
params.max_seq_length = 128
params.seed = 1
params.examplespath = params.virtualenv != "" ? params.virtualenv + "/transformers/examples" : "transformers/examples"
params.converttensor = params.virtualenv != "" ? params.virtualenv + "/rust-bert/target/release/convert-tensor" : "convert-tensor"
params.cache_dir = ""

if (!params.containsKey('name') || !params.containsKey('traindata') || !params.containsKey('testdata') || !params.containsKey('devdata') || !params.containsKey('model')) {

    log.info "Usage:"
    log.info " train_tagger.nf [options]"
    log.info ""
    log.info "Mandatory parameters:"
    log.info "  --name [name] - A name for the resulting model, a directory with this name will be created"
    log.info "  --traindata [file] - The datafile to train on (TSV format, see specification below)"
    log.info "  --devdata [file] - The datafile to run tuning on (TSV format, see specification below)"
    log.info "  --testdata [file] - The datafile to test on (TSV format, see specification below)"
    log.info "  --model [model] - Tranformers pre-trained model name or path (e.g. bert-base-multilingual-cased)"
    log.info "  --examplespath [path] - Path to the transformers examples source code on your system (not necessarily needs to be available on all nodes)"
    log.info "  --converttensor [path] - Full path to the convert-tensor binary form bert-rust (assumes this is available on all nodes)"
    log.info ""
    log.info "Optional parameters:"
    log.info "  --virtualenv [path] - The Python virtual environment to use (autodetected if you run this script from within a virtualenv). Assumes this is available at the same path on every computing node!"
    log.info ""
    log.info "Optional parameters inherited from Transformers' run_ner.py (see there for a description):"
    log.info "  --num_train_epochs, --per_gpu_train_batch_size, --save_steps, --seed, --max_seq_length, --cache_dir"
    log.info ""
    log.info "File format:"
    log.info "  TSV - Tab Separated; one token per line, two columns (token,tag). Empty lines delimit sentences."
    exit 2

}

train = Channel.fromPath(params.traindata)
dev = Channel.fromPath(params.devdata)
test = Channel.fromPath(params.testdata)

process extract_labels {
    input:
    file traindata from train
    file devdata from dev
    file testdata from test

    output:
    file "labels.txt" into labelsfile

    script:
    """
    cat "$traindata" "$devdata" "$testdata" | cut -f 2 | grep -v "^\$" | sort | uniq > labels.txt
    """

}

process run_ner {
    publishDir params.name, pattern: "{pytorch_model.bin,*.json,vocab.txt,*results.txt}", mode: 'copy', overwrite: true

    input:
    file "train.txt" from Channel.fromPath(params.traindata)
    file "dev.txt" from Channel.fromPath(params.devdata)
    file "test.txt" from Channel.fromPath(params.testdata)
    file labels from labelsfile
    file run_ner_script from Channel.fromPath(params.examplespath + "/ner/run_ner.py")
    file ner_utils_script from Channel.fromPath(params.examplespath + "/ner/utils_ner.py")
    val model from params.model
    val epochs from params.num_train_epochs
    val batch_size from params.per_gpu_train_batch_size
    val seed from params.seed
    val save_steps from params.save_steps
    val cache_dir from params.cache_dir
    val virtualenv from params.virtualenv

    output:
    file "pytorch_model.bin" into pytorch_model
    file "config.json" into configfile
    file "vocab.txt" into vocabfile
    file "eval_results.txt" into dev_results
    file "test_results.txt" into test_results

    script:
    """
    set +u
    if [ ! -z "${virtualenv}" ]; then
        source ${virtualenv}/bin/activate
    fi
    set -u

    if [ ! -z "$cache_dir" ]; then
        extra="--cache_dir=${cache_dir}"
    else:
        extra=""
    fi

    python3 $run_ner_script \$extra --data_dir ./ --output_dir ./ --labels $labels --model_name_or_path $model --num_train_epochs $epochs --seed $seed --per_gpu_train_batch_size $batch_size --save_steps $save_steps --do_train --do_eval --do_predict
    exit \$?
    """
}

process convert_to_npz {
    input:
    file model from pytorch_model
    val virtualenv from params.virtualenv

    output:
    file "model.npz" into npz_model

    script:
    """
    #!${virtualenv}/bin/python

    weights = torch.load("$model", map_location='cpu')
    nps = {}
    for k, v in weights.items():
        k = k.replace("gamma", "weight").replace("beta", "bias")
        nps[k] = np.ascontiguousarray(v.cpu().numpy())

    np.savez('model.npz', **nps)
    """
}

process convert_tensor {
    publishDir params.name, pattern: "*.ot", mode: 'copy', overwrite: true

    input:
    file model from npz_model
    val virtualenv from params.virtualenv
    val converttensor from params.converttensor

    output:
    file "model.ot" into tensor_model

    script:
    """
    set +u
    if [ ! -z "${virtualenv}" ]; then
        source ${virtualenv}/bin/activate
    fi
    set -u

    $converttensor $model model.ot
    exit \$?
    """
}


tensor_model.subscribe { println "Final output model written to " +  params.name + '/' + it.name }
