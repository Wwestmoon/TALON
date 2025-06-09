'''
Description: This file contains the utility functions for the OpenAI API.
Author: Don
'''
import time
from openai import OpenAI
import regex as re

# new
api_key = ""
# old
#api_key = ""
base_url=""
# openai.api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(
    base_url=base_url,
    api_key=api_key
)

def llm(prompt, model, temperature=0, stop=None, logprobs=None, n=1, max_tokens=512, verbose=False, history = None):

    for trial in range(3):
        if trial > 0:
            time.sleep(trial * 2)
        try:
            
            messages = [{
                "role": "system",
                "content": "You are a helpful assistant."
            }]
            # if history:
            #     for h in history:
            #         messages.append({"role": h['role'], "content": h['content']})
            messages.append({"role": "user", "content": prompt})
            # set parameters
            parameters = {
                "messages": messages,
                "model": model,
                "max_tokens": max_tokens,
                "stop": stop,
                "temperature": temperature,
                "n": n,
                "logprobs": logprobs
            }
            #print(">" * 20, "LLMs parameters", parameters)
            parameters.update({"messages": messages})
            resp = client.chat.completions.create(**parameters)
            # if verbose:
            #     #print(resp)
            #     continue
            text = resp.choices[0].message.content
            assert len(text) > 0
        except BaseException as e:
            print(">" * 20, "LLMs error", e)
            resp = None
        if resp is not None:
            break
    return resp

def generate_result_openai(prompt, model_name, temperature, max_new_tokens, generation_num, history=None):
    results = []
    cot_results= []
    response = llm(prompt, model_name, stop="```\n", max_tokens=max_new_tokens, verbose=True, temperature=temperature, n = generation_num, history = history)
    for i in range(generation_num):
        # response = llm(prompt, model_name, stop="", max_tokens=max_new_tokens, verbose=True, temperature=temperature)
        result, token = process_result_openai(response, i)
        cot_results.append({"gen{}".format(str(i + 1)):result})
        content_match = re.search(r'\[\[(.*?)\]\]', result)
        content_result = content_match.group(1) if content_match else None
        results.append({"gen{}".format(str(i + 1)):str(content_result)})
    return results, cot_results, token

def process_result_openai(response, index=0):
    try:
        res = response.choices[index].message.content
    except:
        res = "ERROR[[ERROR]]"
    return res, token_count(response)
def token_count(response):
    try:
        a = {}
        a['total_tokens'] = response.usage.total_tokens
        a['completion_tokens'] = response.usage.completion_tokens
        a['prompt_tokens'] = response.usage.prompt_tokens
        return a
    except:
        return 0
def _test_llm():
    model = "gemini-1.5-flash-002"

    prompt = "Q: American Callan Pinckney’s eponymously named system became a best-selling (1980s-2000s) book/video franchise in what genre?"
    prompt += "A: "
    #prompt = 'hello'
    res = llm(prompt, model, stop="", max_tokens=2000, verbose=True, temperature=0.7, n = 5)
    #print(res)
    print(res)

if __name__ == "__main__":
    _test_llm()
   