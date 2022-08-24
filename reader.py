from typing import Dict, Optional

from allennlp.common.tqdm import Tqdm
from transformers import AutoTokenizer

import json


class Roberta_Reader():

    def __init__(self):
        max_pieces = 512
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', max_length=max_pieces)

    def read(self, filepath):

        data_file = open(filepath, 'r')

        item_jsons = []
        for line in data_file:
            item_jsons.append(json.loads(line.strip()))

        for item_json in Tqdm.tqdm(item_jsons, total=len(item_jsons)):
            # self._debug_prints -= 1
            # if self._debug_prints >= 0:
            #     logger.info(f"====================================")
            #     logger.info(f"Input json: {item_json}")
            item_id = item_json["id"]

            statement_text = item_json["phrase"]
            metadata = {} if "metadata" not in item_json else item_json["metadata"]
            context = item_json["context"] if "context" in item_json else None

            yield {'item_id':item_id,
                   'question':statement_text,
                    'answer_id':item_json["answer"],
                    'context': context}

    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         question: str,
                         answer_id: int = None,
                         context: str = None,
                         org_metadata: dict = {}) -> Dict:


        encoded_input = self.tokenizer(question, context)
        print(self.tokenizer.decode(encoded_input["input_ids"]))
        return {
            'id': item_id ,
            'phrase': encoded_input,
            'label': answer_id,
        }
