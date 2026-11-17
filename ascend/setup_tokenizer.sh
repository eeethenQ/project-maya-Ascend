#!/bin/sh
# ascend/setup_tokenizer.sh - assemble the tokenizer directory serve/server.py needs for this checkpoint.
#
# The W8A8 export ships no tokenizer at all, and serve/server.py wants a directory holding tokenizer.json
# plus chat_template.jinja (it treats a template containing "｜DSML｜" as DeepSeek's - serve/frontend.py:59).
# V4.1 and V4 share the vocabulary (129280, bos 0, eos 1), so the V4 export's tokenizer.json serves; the
# template is this repository's own DeepSeek-V4.1 one (tools/ds4/chat_template.jinja).
#
#   ascend/setup_tokenizer.sh \
#       /mnt/weights/DeepSeek-V4-Flash-0731-w8a8 \
#       /home/q00999609/dsv41-tokenizer
set -e
src=$1
out=$2
here=$(cd "$(dirname "$0")" && pwd)
if [ -z "$src" ] || [ -z "$out" ]; then
    echo "usage: $0 <dir with tokenizer.json> <new tokenizer dir>" >&2
    exit 2
fi
mkdir -p "$out"
ln -sf "$src/tokenizer.json" "$out/tokenizer.json"
[ -f "$src/tokenizer_config.json" ] && ln -sf "$src/tokenizer_config.json" "$out/tokenizer_config.json"
cp "$here/../tools/ds4/chat_template.jinja" "$out/chat_template.jinja"
echo "tokenizer directory: $out"
ls -l "$out"
