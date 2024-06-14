# GraphCoder: Enhancing Repository-Level Code Completion via Coarse-to-fine Retrieval Based on Code Context Graph

## Overview

In this paper, we propose GraphCoder, a retrieval-augmented code completion framework that leverages LLMs' general code knowledge and the repository-specific knowledge via a graph-based retrieval-generation process. In particular, GraphCoder captures the context of completion target through code context graph (CCG) that consists of control-flow and data/control-dependence between code statements, a more structured way to capture the completion target context than the sequence-based context used in existing retrieval-augmented approaches; based on CCG, GraphCoder further employs a coarse-to-fine retrieval process to locate context-similar code snippets with the completion target from the current repository.

## Project Structure

The structure of this project is shown as follows:

```
├─ RepoEval-Updated    # Input dataset for code completion tasks
    ├─ api_level.python.test.jsonl
    ├─ api_level.java.test.jsonl
    ├─ line_level.python.test.jsonl
    └─ line_level.java.test.jsonl
├─ repositories    # The original repositories that RepoEval-Updated built from
    ├─ devchat
    ├─ nemo_aligner
    ├─ awslabs_fortuna
    ├─ task_weaver
    ├─ huggingface_diffusers
    ├─ opendilab_ACE
    ├─ metagpt
    ├─ nerfstudio-project_nerfstudio
    ├─ axlearn
    ├─ adalora
    ├─ chatgpt4j
    ├─ rusty-connector
    ├─ neogradle
    ├─ mybatis-flex
    ├─ rocketmq
    ├─ harmonic-hn
    ├─ open-dbt
    ├─ custom-pixel-dungeon
    ├─ cms-oss
    └─ min
├─ tree-sitter-python    # The dependence folder for code context graph generation (tree-sitter parser for python)
├─ tree-sitter-java      # The dependence folder for code context graph generation (tree-sitter parser for java)
├─ utils
    ├─ __init__.py
    ├─ ccg.py    # Generate a code context graph (CCG) from Python code snippets
    ├─ slicing.py    # Generate CCG skuce and its corresponding contet sequence slice
    ├─ build_prompt.py    # Construct prompt with/without retrieval code snippets for code LLMs
    ├─ metrics.py    # Metric calculation for evaluating retrieval and generation effectiveness
    └─ utils.py    # Other tools for tokenizer and file reading/writing
├─ build_graph_database.py    # Build a key-value database for retrieval
├─ build_query_graph.py    # Generate sliced query CCG
├─ search_code.py    # Coarse-to-fine code retrieval
├─ generate_response.py    # Generate the predicted statement based on retrieval results
├─ requirements.txt    # List of packages required to run GraphCoder
└─ my-languages.so    # Dependence file for code context graph generation (tree-sitter parser for python)
```

## Quick Start

#### Step 1: Install Requirements

```
pip install -r requirements.txt
```

The generation of code context graph is based on tree-sitter
```
git clone https://github.com/tree-sitter/tree-sitter-python
```
#### Step 2: Database Construction

```
python build_graph_database.py
```

### Step 3: Code Retrieval

There are 3 input arguments for code retrieval step:

  - query_cases: 
    - `api_level`: Run code retrieval for api-level code completion tasks.
    - `line_level`: Run code retrieval for line-level code completion tasks.

  - mode:
    - `coarse2fine`: This mode corresponds to the code retrieval method in GraphCoder, which performs coarse-grained retrieval and fine-grained re-ranking.
    - `coarse`: This mode corresponds to the variant of GraphCoder, namely GraphCoder-C, which only performs coarse-grained retrieval.
    - `fine`: This mode corresponds to the variant of GraphCoder, namely GraphCoder-F, which performs retrieval only by the fine-grained graph measure (i.e., decay-with-distance subgraph edit distance).

  - gamma: The decay-with-distance factor used in fine-grained step.
    
An example for running code retrieval

```
python search_code.py --query_cases api_level --mode coarse2fine --gamma 0.1
```

### Step 4: Code Generation

There are 5 input arguments for code generation step:

  - input_file_name: Input code completion task with/without retrieval results

  - model: Generation models used in our experiments, including gpt-3.5-turbo-instruct, starcoder(15B), codegen2-16b, codegen2-7b, codegen2-1b

  - mode:
    - `infile`: Generation without retrieval
    - `retrieval`: Generation with retrieval

  - max_top_k: The maximum number of retrieved code snippets

  - max_new_tokens: The maximum number of tokens in the generated completion

    
An example for running code generation

```
python generate_response.py --input_file_name api_level.coarse2fine.10 --model gpt-3.5-turbo-instruct --mode retrieval --max_top_k 10 --max_new_tokens 100
```
