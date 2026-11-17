#!/bin/sh
# ascend/setup_tokenizer.sh - assemble the tokenizer directory serve/server.py needs for this checkpoint.
#
# The W8A8 export ships no tokenizer at all.  V4.1 and V4 share the vocabulary (129280 ids, bos 0, eos 1),
# so the V4 export's tokenizer.json serves; ascend/make_tokenizer.py converts it to the pack layout the
# serve layer reads (vocab.json, merges.txt, and the `deepseek3` pre-tokenizer), copies this repository's
# DeepSeek-V4.1 chat template (tools/ds4/chat_template.jinja - it has to contain the DSML tag for the server
# to treat the model as DeepSeek's, serve/frontend.py:59), and -- with --check -- proves the conversion by
# encoding a corpus with both tokenizers.
#
#   ascend/setup_tokenizer.sh /mnt/weights/DeepSeek-V4-Flash-0731-w8a8 .ascend-tokenizer
#
# Write it *inside the repository* (or any directory the engine container mounts): the engine runs in
# a container, and a path outside its mounts is a different, empty directory there.
set -e
src=$1
out=$2
here=$(cd "$(dirname "$0")" && pwd)
if [ -z "$src" ] || [ -z "$out" ]; then
    echo "usage: $0 <dir with tokenizer.json> <new tokenizer dir>" >&2
    exit 2
fi
exec "${PYTHON:-python3}" -m ascend.make_tokenizer --hf "$src" --out "$out" \
    --template "$here/../tools/ds4/chat_template.jinja" --check
