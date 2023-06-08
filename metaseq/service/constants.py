# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

MAX_SEQ_LEN = 64
BATCH_SIZE = 1024  # silly high bc we dynamically batch by MAX_BATCH_TOKENS
MAX_BATCH_TOKENS = 1024 * 64
DEFAULT_PORT = 6010
MODEL_PARALLEL = 1
TOTAL_WORLD_SIZE = 1
MAX_BEAM = 1
SAMPLING_TOPP = 1
TEMPERATURE = 0.8
GEN_LEN = 40_000_000
LOGPROBS = 0
DESCRIPTION = f"temp_{TEMPERATURE}"
MOL_REPR = "selfies"  # selfies/smiles
CHECKPOINT_ITER = 190000
CHECKPOINT_FOLDER = "../Molecular_Generation_with_GDB13/Checkpoints/Sas_3_selfies"  # Generations_aspirin_0.4 / Generations_sas_3_selfies / Generations_all

# TOKENIZER = "--hf-tokenizer Molecular_Generation_with_GDB13/Data/tokenizers/tokenizer_smiles/old_tokenizer_smiles.json" \
TOKENIZER = (
    "--vocab-filename ../Molecular_Generation_with_GDB13/Data/tokenizers/tokenizer_smiles/vocab.txt"
    if MOL_REPR == "smiles"
    else "--hf-tokenizer  ../Molecular_Generation_with_GDB13/Data/tokenizers/tokenizer_selfies/tokenizer.json"
)

# TOKENIZER = "data-bin/deepchem_dir/vocab.txt" if MOL_REPR=="smiles" else "data-bin/tokenizers/tokenizer_selfies/tokenizer.json"

# tokenizer files
MODEL_FILE = os.path.join(CHECKPOINT_FOLDER, f"checkpoint_{CHECKPOINT_ITER}.pt")
FILE_PATH = f"./{CHECKPOINT_FOLDER}/OPT_iter_{CHECKPOINT_ITER}_{DESCRIPTION}.csv"


LAUNCH_ARGS = [
    "--task language_modeling",
    TOKENIZER,
    "--bpe hf_byte_bpe",  # ????
    f"--path {MODEL_FILE}",
    f"--beam {MAX_BEAM}",
    f"--sampling-topp {SAMPLING_TOPP}",
    f"--logprobs {LOGPROBS}",
    f"--generation-len {GEN_LEN}",
    f"--temperature {TEMPERATURE}",
    f"--description {DESCRIPTION}",
    f"--output-file-path {FILE_PATH}",
    f"--mol-repr {MOL_REPR}",
    "--sampling",
    "--distributed-port 13000",
    "--checkpoint-shard-count 1",
    f"--batch-size {BATCH_SIZE}",
    f"--buffer-size {BATCH_SIZE * MAX_SEQ_LEN}",
    f"--max-tokens {BATCH_SIZE * MAX_SEQ_LEN}",
    "/tmp",  # required "data" argument.
]

# Optional arg overrides which influence model loading during inference
INFERENCE_ARG_OVERRIDES = {}
