import os
import re
import sys
import pandas as pd
import json
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import logging
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)


def _bioclean(caption):
    return re.sub('[.,?;*!%^&_+():-\[\]{}]',
                  '',
                  caption.replace('"', '')
                  .replace('/', '')
                  .replace('\\', '')
                  .replace("'", '')
                  .strip()
                  .lower())


class CaptionsEvaluation:

    def __init__(self, gold_dir, results_dir):
        self.results_dir = results_dir
        self.gold_dir = gold_dir
        self.gold_data = {}
        self.result_data = {}
        self.result_data = {}

    def _load_data(self):
        gold_csv = pd.read_csv(self.gold_dir, sep="\t", header=None, names=["unique_ids", "image_ids", "captions", "labels", "image_codes"], dtype=object)
        gold_csv = gold_csv[gold_csv.columns[1:3]]
        self.gold_data = dict(zip(gold_csv.image_ids, gold_csv.captions))
        with open(self.results_dir) as json_file:
            result_data_list = json.load(json_file)
        for item in result_data_list:
            self.result_data[item['image_id']] = item['caption']

    def _preprocess_captions(self, images_caption_dict):
        """
        :param images_caption_dict: Dictionary with image ids as keys and captions as values
        :return: Dictionary with the processed captions as values and the id of the images as
        key
        """
        processed_captions_dict = {}
        for image_id in images_caption_dict:
            processed_captions_dict[image_id] = [_bioclean(images_caption_dict[image_id])]
        return processed_captions_dict

    def compute_ms_coco(self):
        """Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)
        :param gts: Dictionary with the image ids and their gold captions,
        :param res: Dictionary with the image ids and their generated captions
        :print: Evaluation score (the mean of the scores of all the instances) for each measure
        """

        # load the csv files, containing the results and gold data.
        logging.info("Loading data")
        self._load_data()

        # Preprocess captions
        logging.info("Preprocessing captions")
        self.gold_data = self._preprocess_captions(self.gold_data)
        self.result_data = self._preprocess_captions(self.result_data)
        if len(self.gold_data) == len(self.result_data):
            # Set up scorers
            scorers = [
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr")
            ]

            # Compute score for each metric
            logging.info("Computing COCO score.")
            for scorer, method in scorers:
                print("Computing", scorer.method(), "...")
                score, scores = scorer.compute_score(self.gold_data, self.result_data)
                if type(method) == list:
                    for sc, m in zip(score, method):
                        print("%s : %0.3f" % (m, sc))
                else:
                    print("%s : %0.3f" % (method, score))
        else:
            logging.info("Gold data len={0} and results data len={1} have not equal size".format(len(self.gold_data), len(self.result_data)))

if __name__ == "__main__":
    if len(sys.argv) == 3:
        evaluation = CaptionsEvaluation(sys.argv[1], sys.argv[2])
        evaluation.compute_ms_coco()
    elif len(sys.argv) == 4:
        evaluation = CaptionsEvaluation(sys.argv[1], sys.argv[2], sys.argv[3])
        evaluation.compute_ms_coco()
    else:
        raise NotImplementedError