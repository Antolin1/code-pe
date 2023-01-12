#!/bin/bash

mkdir grammars
cd grammars
git clone https://github.com/tree-sitter/tree-sitter-python.git
cd ..
python build_grammars.py
