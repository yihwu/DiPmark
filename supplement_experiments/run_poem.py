import os

os.environ['TRANSFORMERS_CACHE'] = '/fs/nexus-scratch/ywu42/.cache'

import poem_generation as poem
import text_summarization as ts

def main():
    ts.get_output.pipeline()
    ts.evaluate_ppl.pipeline()

if __name__ == "__main__":
    main()

