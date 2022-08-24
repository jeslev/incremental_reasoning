import json, random, pickle, os
import numpy as np

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel

from reader import Roberta_Reader
import logger

log = logger.get_logger(__name__)


class Winogrande():
    '''
    Algorithm Winogrande

    1. Train a LM model (Roberta) with small training set (ex: 6k, 5k training, 1k validation)
    2. Use this fine-tuned models (Rob_emb) to compute the input embeddings [CLS_token]
    3. Remove the elements from the training at (1) in the dataset.
    4. Use this representation for the iterative part
    '''
    def __init__(self,cfg):
        self._cfg = cfg
        self._filtered_dataset = None

        self._embeddings_path = self._cfg.get("setting/system/output_embeddings")
        self._embeddings_path = os.path.join(self._embeddings_path, cfg.name+".pkl")
        self._embeddings = None
        if os.path.isfile(self._embeddings_path):
            with open(self._embeddings_path, "rb") as file:
                self._embeddings = pickle.load(file)
        else:
            self.preprocess_input()


    def preprocess_input(self):

        # Create triples (label, input, cls token representation)
        filepath = self._cfg.get("setting/system/inputfile")

        triples_list = list()
        model_str = self._cfg.get("setting/system/model")
        sample = self._cfg.get("setting/system/sample")

        reader = Roberta_Reader()
        tokenizer = RobertaTokenizer.from_pretrained(model_str, max_length=256)
        model = RobertaModel.from_pretrained(model_str)

        log.debug("Processing dataset")
        if type(filepath) is not list:
            filepath = [filepath]

        # Iterate over different datasets
        for id_ds, _fpath in enumerate(filepath):
            for i,instance in enumerate(reader.read(_fpath)):
                #print("Instance ",i)
                #print(instance)
                encoded_input = tokenizer(instance["question"], instance["context"], return_tensors="pt")
                #print(encoded_input)
                output = model(**encoded_input)
                cls_token = output.pooler_output.detach().numpy()
                #print(cls_token)

                element = { "id": instance["item_id"], "cls": cls_token, "label": instance["answer_id"], "ds": id_ds}
                triples_list.append(element)

                if sample != -1 and i>sample:
                    break

        log.debug("Original dataset has " + str(len(triples_list)) + " entries.")


        with open(self._embeddings_path ,"wb") as file:
            pickle.dump(triples_list, file)
        log.debug("Embedding saved into " + self._embeddings_path)
        self._embeddings = triples_list


    def run(self, ensemble_size = 2, training_set_size = 10, cutoff_size = 300, threshold = 0.75):
        '''
        From Winogrande paper they used
        :param data_orig:
        :param ensemble_size: # of classifier (64)
        :param training_set_size:  training size for the classifiers (10000)
        :param cutoff_size: top-k(500)
        :param threshold: 0.75
        :return: Cleaned dataset
        '''

        dataset = self._embeddings.copy() # list( {id, label, cls} )
        n_iteration = 0
        while len(dataset) > training_set_size:
            n_iteration += 1
            log.debug("Iteration " + str(n_iteration))
            # get all labels as list
            all_labels = []
            for entry in dataset:
                all_labels.append(entry["label"])

            # Filtering phase
            # Initialize counter
            E = dict()


            # Train multiple classifiers
            random_seeds = random.sample(range(1,1024), ensemble_size)# Generate n seeds
            #print(random_seeds)
            for i in range(ensemble_size):
                # Select subset for train/validation, using 4/5 for training 1/5 for test
                X_train, X_test, y_train, y_test = \
                    train_test_split(dataset, all_labels, train_size=training_set_size,
                                     test_size=len(dataset)-training_set_size, random_state=random_seeds[i])

                log.debug("Using " + str(len(y_train)) + " for training, and " + str(len(y_test)) + " for test.")
                predictions = self.train_classifier(X_train, X_test, y_train, y_test)

                # Update E counter
                for entry, prediction in zip(X_test, predictions):
                    _id = entry["id"]
                    current_predictons = E.get(_id,[])
                    if prediction == entry["label"]: # correct
                        current_predictons.append(1)
                    else: # wrong
                        current_predictons.append(0)
                    E[_id] = current_predictons

            # Compute score for all dataset
            score = self.compute_score(dataset, E)

            # select top-k
            S = self.compute_topk(score, threshold, cutoff_size) # elements to remove
            log.debug("Selected " + str(len(S)) + " elements to remove.")

            # remove S from dataset
            dataset = self.remove_dataset(dataset, S)

            # if elements to remove < k then break
            if len(S) < cutoff_size:
                break

        self._filtered_dataset = dataset


    def compute_score(self, dataset, E):
         # compute the ratio of correct predictions
        score = list()
        for entry in dataset:
            _id = entry["id"]
            matches = E.get(_id, [])
            if len(matches) > 0:
                _score = float(1.0 * np.array(matches).sum() / len(matches) )
            else:
                _score = 0
            score.append((_id,_score))
        return score


    def compute_topk(self, score, threshold, cutoff_size):

        ranked_inputs = sorted(score, key=lambda tup: tup[1], reverse=True) # Sort
        # Select those above threshold
        to_remove = []
        list_scores = []
        for el in ranked_inputs:
            _id, _score = el
            if _score >= threshold:
                to_remove.append(_id)
                list_scores.append(_score)
            else:
                break
        # Select top-k
        log.debug("Threshold " + str(threshold))
        for i in range(min(10, len(list_scores))):
            log.debug(str(list_scores[i]))
        if len(to_remove) > cutoff_size:
            to_remove = to_remove[:cutoff_size]
        return to_remove


    def remove_dataset(self, dataset, to_remove):
        tmp_dataset = []
        for entry in dataset:
            if entry["id"] in to_remove:
                continue
            tmp_dataset.append(entry)
        return tmp_dataset

    def train_classifier(self, X_train, X_test, y_train, y_test):

        # Train classifier with subset
        lr_clf = LogisticRegression(max_iter=400)

        train_features = [entry["cls"] for entry in X_train]
        train_features = np.concatenate(train_features, axis=0)
        #print(train_features.shape)
        lr_clf.fit(train_features, y_train)

        test_features = [entry["cls"] for entry in X_test]
        test_features = np.concatenate(test_features, axis=0)
        #print(test_features.shape)
        #acc = lr_clf.score(test_features, y_test)
        predictions = lr_clf.predict(test_features)

        return predictions


    def save(self, output_path):

        filepaths = self._cfg.get("setting/system/inputfile")
        reader = Roberta_Reader()

        list_kept_ids = {entry["id"]:entry["ds"] for entry in self._filtered_dataset}
        #dataset_id = [entry["ds"] for entry in self._filtered_dataset]
        _ids = list(list_kept_ids.keys())


        if type(filepaths) is not list:
            filepaths = [filepaths]

        for id_ds, filepath in enumerate(filepaths):
            filtered_dataset = []
            for i,instance in enumerate(reader.read(filepath)):
                _ds = list_kept_ids.get(instance["item_id"], -1)
                if _ds==-1: #Not in list
                    continue
                if _ds == id_ds: # is the current dataset, write
                    changed_key = {"id":instance["item_id"],
                     "phrase":instance["question"],
                     "answer":instance["answer_id"],
                     "contesxt":instance["context"],
                     "metadata":instance.get("org_metadata",{})}
                    filtered_dataset.append(changed_key)

            # Write file
            Path(output_path).mkdir(parents=True, exist_ok=True)

            old_dataset_name = filepath.split("/")[-1]
            target_file = os.path.join(output_path, old_dataset_name)
            with open(target_file, "w") as file:
                for line in filtered_dataset:
                    file.write(json.dumps(line)+"\n")

