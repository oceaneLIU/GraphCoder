import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
import openai
from utils.utils import load_jsonl, dump_jsonl, make_needed_dir
import copy
from utils.build_prompt import build_prompt
from utils.utils import CodexTokenizer, CodeGenTokenizer, StarCoderTokenizer

device = "cuda"


def parser_args():

    parser = argparse.ArgumentParser(description="Generate response from llm")
    parser.add_argument('--input_file_name', default='api_level', type=str)
    parser.add_argument('--model', default='gpt-3.5-turbo-instruct', type=str)
    parser.add_argument('--mode', default='retrieval', type=str, choices=['infile', 'retrieval'])
    parser.add_argument('--max_top_k', default=10, type=int)
    parser.add_argument('--max_new_tokens', default=100, type=int)

    return parser.parse_args()


def main(args, input_cases, responses_save_name):

    # set model and tokenizer
    if args.model == 'gpt-3.5-turbo-instruct':
        model = openai.OpenAI(api_key="")
        tokenizer = CodexTokenizer()
        max_num_tokens = 4096
    elif args.model == 'starcoder':
        model = AutoModelForCausalLM.from_pretrained(f"./models_cache/{args.model}/")
        tokenizer_raw = AutoTokenizer.from_pretrained(f"./models_cache/{args.model}-tokenizer/", trust_remote_code=True)
        tokenizer = StarCoderTokenizer(tokenizer_raw)
        max_num_tokens = 8192
        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer_raw.eos_token_id,
            temperature=0,
            pad_token_id=tokenizer_raw.pad_token_id,
        )
    elif args.model in ['codegen2-16b', 'codegen2-7b', 'codegen2-1b']:
        model = AutoModelForCausalLM.from_pretrained(f"{args.model}/")
        tokenizer_raw = AutoTokenizer.from_pretrained(f"./models_cache/{args.model}-tokenizer/", trust_remote_code=True)
        tokenizer = CodeGenTokenizer(tokenizer_raw)
        max_num_tokens = 2048
        generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer_raw.eos_token_id,
            temperature=0,
            pad_token_id=tokenizer_raw.pad_token_id,
        )
    print('Model loading finished')

    # generate response and save
    responses = []
    max_prompt_tokens = max_num_tokens - args.max_new_tokens
    with tqdm(total=len(input_cases)) as pbar:
        for case in input_cases:
            pbar.set_description(f'Processing...')

            prompt = build_prompt(case, tokenizer, max_prompt_tokens, max_top_k=args.max_top_k, mode=args.mode)

            if args.model == 'gpt-3.5-turbo-instruct':
                completion = model.completions.create(
                    model=args.model,
                    prompt=prompt,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature
                )
                response = completion.choices[0].text
            elif args.model == "starcoder":
                prompt_ids = tokenizer_raw(prompt, return_tensors="pt").to(device)
                response_ids = model.generate(prompt_ids['input_ids'],
                                              generation_config=generation_config,
                                              attention_mask=prompt_ids['attention_mask'])
                response = tokenizer.decode(response_ids[0])
                prompt_lines = prompt.splitlines(keepends=True)
                n_prompt_lines = len(prompt_lines)
                response_lines = response.splitlines(keepends=True)
                response = "".join(response_lines[n_prompt_lines:])
            elif args.model in ['codegen2-16b', 'codegen2-7b', 'codegen2-1b']:
                prompt_ids = tokenizer_raw(prompt, return_tensors="pt").to(device)
                response_ids = model.generate(prompt_ids['input_ids'],
                                              generation_config=generation_config,
                                              attention_mask=prompt_ids['attention_mask'])
                response = tokenizer.decode(response_ids[0])
                prompt_lines = prompt.splitlines(keepends=True)
                n_prompt_lines = len(prompt_lines)
                response_lines = response.splitlines(keepends=True)
                response = "".join(response_lines[n_prompt_lines:])
            case_res = copy.deepcopy(case)
            case_res['generate_response'] = response
            responses.append(case_res)
            pbar.update(1)

    dump_jsonl(responses, responses_save_name)


if __name__ == "__main__":
    args = parser_args()

    # load input
    input_cases = load_jsonl(f"./search_res/{args.input_file_name}.search_res.jsonl")
    print('Input loading finished')

    responses_save_name = f"./generation_results/{args.model}/{args.input_file_name}.{args.mode}.{args.model}.gen_res.jsonl"
    make_needed_dir(responses_save_name)
    main(args, input_cases, responses_save_name)


