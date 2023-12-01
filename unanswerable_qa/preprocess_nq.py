# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Utilities to simplify the canonical NQ data.
The canonical NQ data contains the HTML of each Wikipedia page along with a
sequence of tokens on that page, each of which is indexed into the HTML.
Many users will not want to use the HTML at all, and this file provides
utilities to extract only the text into a new record of the form:
  {
    "example_id": 3902,
    "document_url": "http://wikipedia.org/en/strings"
    "question_text": "what is a string",
    "document_text": "<P> A string is a list of characters in order . </P>",
    "annotations": [{
      "long_answer": { "start_token": 0, "end_token": 12 },
      "short_answers": [{ "start_token": 5, "end_token": 8 }],
      "yes_no_answer": "NONE",
    }],
    "long_answer_candidates": [
      {"start_token": 0, "end_token": 12, "top_level": True}
    ]
  }
which leads to a much smaller training set (4.4Gb instead of 41Gb).
In this representation, the [start, end) indices are into the blank separated
sequence of tokens. So, answer spans can be extracted using the following
snippet:
  " ".join(example["document_text"].split(" ")[`start_token`:`end_token`]).
WARNING: Use `split(" ")` instead of `split()` to avoid complications from
  characters such as `\u180e` which may or may not be recognized as a whitespace
  character depending on your python version.
To avoid complications at test time, we do not provide a simplified version
of the development data, and there is no simplified version of the hidden test
set. If you rely on the simplified data, then you must call the
`simplify_nq_example` function below on every example that is passed in at test
time.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jsonlines
import random
from collections import Counter

import re

import json
from tqdm import tqdm


def get_nq_tokens(simplified_nq_example):
    """Returns list of blank separated tokens."""

    if "document_text" not in simplified_nq_example:
        raise ValueError("`get_nq_tokens` should be called on a simplified NQ"
                         "example that contains the `document_text` field.")

    return simplified_nq_example["document_text"].split(" ")


def simplify_nq_example(nq_example):
    r"""Returns dictionary with blank separated tokens in `document_text` field.
    Removes byte offsets from annotations, and removes `document_html` and
    `document_tokens` fields. All annotations in the ouput are represented as
    [start_token, end_token) offsets into the blank separated tokens in the
    `document_text` field.
    WARNING: Tokens are separated by a single blank character. Do not split on
      arbitrary whitespace since different implementations have different
      treatments of some unicode characters such as \u180e.
    Args:
      nq_example: Dictionary containing original NQ example fields.
    Returns:
      Dictionary containing `document_text` field, not containing
      `document_tokens` or `document_html`, and with all annotations represented
      as [`start_token`, `end_token`) offsets into the space separated sequence.
    """

    def _clean_token(token):
        """Returns token in which blanks are replaced with underscores.
        HTML table cell openers may contain blanks if they span multiple columns.
        There are also a very few unicode characters that are prepended with blanks.
        Args:
          token: Dictionary representation of token in original NQ format.
        Returns:
          String token.
        """
        return re.sub(u" ", "_", token["token"])

    text = " ".join([_clean_token(t) for t in nq_example["document_tokens"]])

    def _remove_html_byte_offsets(span):
        if "start_byte" in span:
            del span["start_byte"]

        if "end_byte" in span:
            del span["end_byte"]

        return span

    def _clean_annotation(annotation):
        annotation["long_answer"] = _remove_html_byte_offsets(
            annotation["long_answer"])
        annotation["short_answers"] = [
            _remove_html_byte_offsets(sa) for sa in annotation["short_answers"]
        ]
        return annotation

    simplified_nq_example = {
        "document_title": nq_example["document_title"],
        "question_text": nq_example["question_text"],
        "example_id": nq_example["example_id"],
        "document_url": nq_example["document_url"],
        "document_text": text,
        "long_answer_candidates": [
            _remove_html_byte_offsets(c)
            for c in nq_example["long_answer_candidates"]
        ],
        "annotations": [_clean_annotation(a) for a in nq_example["annotations"]]
    }

    if len(get_nq_tokens(simplified_nq_example)) != len(
            nq_example["document_tokens"]):
        raise ValueError("Incorrect number of tokens.")

    return simplified_nq_example


def covert_to_squad_format(nq_sample, answerable_threshold=2):
    sample = simplify_nq_example(nq_sample)
    title = sample['document_title']
    sample_id = str(sample['example_id'])
    question = sample['question_text']
    tokens = sample['document_text'].split(" ")

    long_answers = Counter()
    short_answers = Counter()
    has_short_ans_count = 0
    for annotation in sample['annotations']:
        long_start = annotation['long_answer']['start_token']
        long_end = annotation['long_answer']['end_token']
        if long_start >= 0 and long_end >= 0:
            long_answers[(long_start, long_end)] += 1
        if annotation['short_answers']:
            has_short_ans_count += 1
            for short_answer in annotation['short_answers']:
                short_start = short_answer['start_token']
                short_end = short_answer['end_token']
                if short_start >= 0 and short_end >= 0:
                    short_answers[(short_start, short_end)] += 1
    if sum(long_answers.values()) >= answerable_threshold and has_short_ans_count >= answerable_threshold:
        max_short_count = max(short_answers.values())
        most_popular_short_ans = [
            span for span, count in short_answers.items()
            if count == max_short_count
        ]
        selected_short = random.choice(most_popular_short_ans)
        compatible_long_ans = {
            span: count for span, count in long_answers.items()
            if span[0] <= selected_short[0] and span[1] >= selected_short[1]
        }
        max_long_count = max(compatible_long_ans.values())
        most_popular_long_ans = [
            span for span, count in compatible_long_ans.items()
            if count == max_long_count
        ]
        selected_long = random.choice(most_popular_long_ans)
        compatible_short_ans = [
            span for span in short_answers.keys()
            if span[0] >= selected_long[0] and span[1] <= selected_long[1]
        ]
        context = ' '.join(tokens[selected_long[0]:selected_long[1]])
        answers = []
        for short_start, short_end in compatible_short_ans:
            text = ' '.join(tokens[short_start:short_end])
            answer_start = len(' '.join(tokens[selected_long[0]:short_start+1]))\
                - len(tokens[short_start])
            assert context[answer_start:answer_start+len(text)] == text
            answers.append({
                "answer_start": answer_start,
                "text": text,
            })
        return {
            'title': title,
            'paragraphs': [{
                'context': context,
                'qas': [{
                    'question': question,
                    'id': sample_id,
                    'answers': answers,
                    'is_impossible': False
                }]
            }]
        }
    elif sum(long_answers.values()) < answerable_threshold and has_short_ans_count < answerable_threshold:
        random_paragraph = random.choice(sample['long_answer_candidates'])
        context = ' '.join(
            tokens[random_paragraph['start_token']:random_paragraph['end_token']])
        return {
            'title': title,
            'paragraphs': [{
                'context': context,
                'qas': [{
                    'question': question,
                    'id': sample_id,
                    'answers': [],
                    'is_impossible': True
                }]
            }]
        }
    else:
        return None


if __name__ == '__main__':
    random.seed(42)
    data = []
    discarded = 0
    with jsonlines.open('data/nq-dev-all.jsonl', 'r') as nq_dev_all:
        for sample in tqdm(nq_dev_all):
            squad_sample = covert_to_squad_format(sample)
            if squad_sample is not None:
                data.append(squad_sample)
            else:
                discarded += 1
    converted_dataset = {
        'version': 'v1.0',
        'data': data
    }
    with open('data/simplified_nq.json', 'w', encoding='utf8') as output:
        json.dump(converted_dataset, output)
    print(f"Discarded {discarded} samples")
