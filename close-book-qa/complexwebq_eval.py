""" Official evaluation script for v1.0 of the ComplexWebQuestions dataset. """
import argparse
import json
import unicodedata
import re
from numpy.lib.arraysetops import isin
import pandas as pd
import numpy as np

def compare_span_to_answer(spans, answers, question):
    """ Compares one answers to spans, multiple matches are possible
    """
    if len(spans) == 0:
        return []

    found_answers = pd.DataFrame(columns=['span', 'answer', 'span_index'])
    spans_series = pd.Series(spans)
    pre_proc_answers = []
    answers = [answer.lower().strip() for answer in answers]
    for answer in answers:
        proc_answer = unicodedata.normalize('NFKD', answer).encode('ascii', 'ignore').decode(encoding='UTF-8')

        # removing common endings such as "f.c."
        proc_answer = re.sub(r'\W', ' ', proc_answer).lower().strip()
        # removing The, a, an from begining of answer as proposed by SQuAD dataset answer comparison
        if proc_answer.startswith('the '):
            proc_answer = proc_answer[4:]
        if proc_answer.startswith('a '):
            proc_answer = proc_answer[2:]
        if proc_answer.startswith('an '):
            proc_answer = proc_answer[3:]

        pre_proc_answers.append(proc_answer)

    question = question.lower().strip()

    # exact match:
    for pre_proc_answer, answer in zip(pre_proc_answers, answers):

        if answer in spans:
            exact_match_ind = spans.index(answer)
            found_answers = found_answers.append({'span_index': exact_match_ind, 'answer': answer, 'span': answer},
                                                 ignore_index=True)

        if pre_proc_answer in spans:
            exact_match_ind = spans.index(pre_proc_answer)
            found_answers = found_answers.append(
                {'span_index': exact_match_ind, 'answer': answer, 'span': pre_proc_answer}, ignore_index=True)

        # year should match year.
        if question.find('year') > -1:
            year_in_answer = re.search('([1-2][0-9]{3})', answer)
            if year_in_answer is not None:
                year_in_answer = year_in_answer.group(0)

            year_spans = spans_series[spans_series == year_in_answer]
            if len(year_spans) > 0:
                found_answers = found_answers.append(
                    {'span_index': year_spans.index[0], 'answer': answer, 'span': year_in_answer}, ignore_index=True)

    return found_answers.drop_duplicates()

def compute_P1(matched_answers, golden_answer_list, pred_answer):
    P1 = 0
    if len(matched_answers) > 0:
        P1 = 100
    return P1

def evaluate(dataset_df, predictions):
    # please predict the full file
    if len(dataset_df) != len(predictions):
        print('predictions file does not match dataset file number of examples!!!')

    P1 = 0
    for prediction in predictions:

        golden_answer_list = []
        for answer in dataset_df.loc[prediction['ID'],'answers']:
            golden_answer_list.append(answer['answer'])
            golden_answer_list += answer['aliases']

        if not None in golden_answer_list:
            matched_answers = compare_span_to_answer([prediction['answer']], golden_answer_list, dataset_df.loc[prediction['ID'], 'question'])
            curr_P1 =  compute_P1(matched_answers, golden_answer_list, prediction['answer'])
            P1 += curr_P1
    return P1/len(dataset_df)


def complexQ_em(pred, gold, question):
    """
    @xwhan
    pred: predicted answer
    gold: list of 
    """
    assert isinstance(pred, str)
    assert isinstance(gold, list)
    if None in gold:
        return False
    return len(compare_span_to_answer([pred], gold, question)) > 0

def chain_baseline_eval():
    """
    eval zero-shot performance of Asai et al. 2020
    """

    # baseline
    id2answer = json.load(open("/private/home/xwhan/code/learning_to_retrieve_reasoning_paths/results/complexwebq_qa_res.json"))

    # ours
    # id2answer = json.load(open("/private/home/xwhan/data/hotpot/results/complex_dense_qa_res.json"))["answer"]

    raw_dev = [json.loads(l) for l in open("/private/home/xwhan/data/ComplexWebQ/complexwebq_dev_qas.txt").readlines()]
    assert len(id2answer) == len(raw_dev)

    id2q = {_["id"]: _["question"] for _ in raw_dev}
    id2gold = {_["id"]: _["answer"] for _ in raw_dev}

    ems = [complexQ_em(id2answer[id_].lower().strip(), id2gold[id_], id2q[id_]) for id_ in id2answer.keys()]

    print(len(ems))
    print(np.mean(ems))

if __name__ == '__main__':
    #expected_version = '1.0'
    # parser = argparse.ArgumentParser(
    #     description='Evaluation for ComplexWebQuestions ')
    # parser.add_argument('dataset_file', help='Dataset file')
    # parser.add_argument('prediction_file', help='Prediction File')
    # args = parser.parse_args()
    # with open(args.dataset_file) as dataset_file:
    #     dataset_df = pd.DataFrame(json.load(dataset_file)).set_index('ID')
    # with open(args.prediction_file) as prediction_file:
    #     predictions = json.load(prediction_file)
    # print(json.dumps(evaluate(dataset_df, predictions)))

    chain_baseline_eval()