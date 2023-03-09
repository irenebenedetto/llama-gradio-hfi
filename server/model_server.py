import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import argparse
from example import setup_model_parallel, load
import os
import sys


app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s.%(funcName)s() - line: %(lineno)s - %(levelname)s - %(message)s",
)

log = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--model_size", type=str, required=False, default="7B")
parser.add_argument("--max_batch_size", type=int, required=False, default=2)
parser.add_argument("--max_seq_len", type=int, required=False, default=600)
parser.add_argument("--port", type=int, required=False, default=8080)

args = parser.parse_args()
log.info(f"args: {args}")


local_rank, world_size = setup_model_parallel()
if local_rank > 0:
    sys.stdout = open(os.devnull, 'w')

generator = load(
    max_seq_len=args.max_seq_len,
    max_batch_size=args.max_batch_size,
    ckpt_dir=f"./models/{args.model_size}",
    tokenizer_path="./models/tokenizer.model",
    local_rank=local_rank, world_size=world_size
)


def generate_text(text, max_gen_len=128, temperature=0.8, top_p=0.95, postprocess="Raw output"):
    print('Generating text...')
    max_gen_len = int(max_gen_len)
    temperature = temperature / 100
    top_p = top_p / 100
    output_text = generator.generate([text], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)

    if postprocess == "Raw output":
        return output_text[0]
    else:

        output_text = output_text[0].replace(text, '').strip()
        return output_text.split('\n')[0]

class BodyParamsSingleText(BaseModel):
    prompt: str
    max_gen_len: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    postprocessing: str = "Raw output"



@app.post('/inference')
async def inference(input: BodyParamsSingleText):
    logging.info(input)
    output = generate_text(input.prompt, input.max_gen_len, input.temperature, input.top_p, input.postprocessing)
    return output


if __name__ == "__main__":
    uvicorn.run("mock_service.run:app", host="0.0.0.0",port=args.port, log_level="info")


