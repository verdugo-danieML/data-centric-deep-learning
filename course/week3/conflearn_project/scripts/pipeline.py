import os
import torch
import random
import numpy as np
import pandas as pd
from os.path import join
from pprint import pprint
from torch.utils.data import DataLoader, TensorDataset

from metaflow import FlowSpec, step, Parameter
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold

from cleanlab.filter import find_label_issues

from conflearn.system import ReviewDataModule, SentimentClassifierSystem
from conflearn.utils import load_config, to_json
from conflearn.paths import DATA_DIR, LOG_DIR, CONFIG_DIR


class TrainIdentifyReview(FlowSpec):
    r"""A MetaFlow that trains a sentiment classifier on reviews of luxury beauty
    products using PyTorch Lightning, identifies data quality issues using CleanLab, 
    and prepares them for review in LabelStudio.

    Arguments
    ---------
    config (str, default: ./config.py): path to a configuration file
    """
    config_path = Parameter('config', help='path to config file', default='./train.json')

    @step
    def start(self):
        r"""Start node.
        Set random seeds for reproducibility.
        """
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        self.next(self.init_system)

    @step
    def init_system(self):
        r"""Instantiates a data module, pytorch lightning module, 
        and lightning trainer instance.
        """
        # configuration files contain all hyperparameters
        config = load_config(join(CONFIG_DIR, self.config_path))

        # a callback to save best model weights
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.train.ckpt_dir,
            monitor='dev_loss',
            mode='min',    # look for lowest `dev_loss`
            save_top_k=1,  # save top 1 checkpoints
            verbose=True,
        )

        trainer = Trainer(
            max_epochs=config.train.optimizer.max_epochs,
            callbacks=[checkpoint_callback],
        )

        # when we save these objects to a `step`, they will be available
        # for use in the next step, through not steps after.
        self.trainer = trainer
        self.config = config

        self.next(self.train_test)

    @step
    def train_test(self):
        """Calls `fit` on the trainer.
        
        We first train and (offline) evaluate the model to see what 
        performance would be without any improvements to data quality.
        """
        # a data module wraps around training, dev, and test datasets
        dm = ReviewDataModule(self.config)

        # a PyTorch Lightning system wraps around model logic
        system = SentimentClassifierSystem(self.config)

        # Call `fit` on the trainer with `system` and `dm`.
        self.trainer.fit(system, dm)
        self.trainer.test(system, dm, ckpt_path='best')

        # results are saved into the system
        results = system.test_results

        # print results to command line
        pprint(results)

        log_file = join(LOG_DIR, 'baseline.json')
        os.makedirs(LOG_DIR, exist_ok=True)
        to_json(results, log_file)  # save to disk

        self.next(self.crossval)
    
    @step
    def crossval(self):
        r"""Confidence learning requires cross validation to compute 
        out-of-sample probabilities for every element. Each element
        will appear in a single cross validation split exactly once. 
        """
        dm = ReviewDataModule(self.config)
        # combine training and dev datasets
        X = np.concatenate([
        np.asarray(dm.train_dataset.embedding),
        np.asarray(dm.dev_dataset.embedding),
        np.asarray(dm.test_dataset.embedding),
        ])
        y = np.concatenate([
        np.asarray(dm.train_dataset.data.label),
        np.asarray(dm.dev_dataset.data.label),
        np.asarray(dm.test_dataset.data.label),
        ])

        probs = np.zeros(len(X))  # we will fill this in

        kf = KFold(n_splits=3)    # create kfold splits

        for train_index, test_index in kf.split(X):
          # Get train and test slices of X and y
          X_train, X_test = X[train_index], X[test_index]
          y_train, y_test = y[train_index], y[test_index]
        
          # Convert to torch tensors
          X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
          y_train_tensor = torch.tensor(y_train, dtype=torch.long)
          X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
          # Create train/test datasets using tensors
          train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
          test_dataset = TensorDataset(X_test_tensor)
        
          # Create train/test data loaders from datasets
          train_loader = DataLoader(train_dataset, batch_size=self.config.train.optimizer.batch_size, shuffle=True)
          test_loader = DataLoader(test_dataset, batch_size=self.config.train.optimizer.batch_size)
        
          # Create SentimentClassifierSystem
          system = SentimentClassifierSystem(self.config)
        
          # Create Trainer and call fit
          trainer = Trainer(max_epochs=self.config.train.optimizer.max_epochs)
          trainer.fit(system, train_loader)
        
          # Call predict on Trainer and the test data loader
          predictions = trainer.predict(system, test_loader)
        
          # Convert probabilities back to numpy (make sure 1D)
          probs_ = torch.cat(predictions).numpy()
        
          # Check the shape of probs_ and handle accordingly
          if probs_.ndim == 2:
            if probs_.shape[1] == 2:
                probs_ = probs_[:, 1]  # Assuming binary classification, take positive class prob
            elif probs_.shape[1] == 1:
                probs_ = probs_.squeeze()  # If it's already the probability for positive class, just squeeze
            else:
                raise ValueError(f"Unexpected number of columns in predictions: {probs_.shape[1]}")
          elif probs_.ndim != 1:
            raise ValueError(f"Unexpected shape of predictions: {probs_.shape}")
        
          probs[test_index] = probs_

    # create a single dataframe with all input features
        all_df = pd.concat([
        dm.train_dataset.data,
        dm.dev_dataset.data,
        dm.test_dataset.data,
        ])
        all_df = all_df.reset_index(drop=True)
        # add out-of-sample probabilities to the dataframe
        all_df['prob'] = probs

        # save to excel file
        all_df.to_csv(join(DATA_DIR, 'prob.csv'), index=False)

        self.all_df = all_df
        self.next(self.inspect)

    @step
    def inspect(self):
        r"""Use confidence learning over examples to identify labels that 
        likely have issues with the `cleanlab` tool. 
        """
        prob = np.asarray(self.all_df.prob)
        prob = np.stack([1 - prob, prob]).T
    
        # rank label indices by issues
        ranked_label_issues = find_label_issues(
            labels=self.all_df.label.values,
            pred_probs=prob,
            return_indices_ranked_by="self_confidence"
        )

        # save this to class
        self.issues = ranked_label_issues
        print(f'{len(ranked_label_issues)} label issues found.')

        # overwrite label for all the entries in all_df
        for index in self.issues:
            label = self.all_df.loc[index, 'label']
            # we FLIP the label!
            self.all_df.loc[index, 'label'] = 1 - label

        self.next(self.review)

    @step
    def review(self):
        r"""Format the data quality issues found such that they are ready to be 
        imported into LabelStudio. We expect the following format:

        [
          {
            "data": {
              "text": <review text>
            },
            "predictions": [
              {
                "value": {
                  "choices": [
                      "Positive"
                  ]
                },
                "from_name": "sentiment",
                "to_name": "text",
                "type": "choices"
              }
            ]
          }
        ]

        See https://labelstud.io/guide/predictions.html#Import-pre-annotations-for-text.and
        """
        outputs = []
        for index in self.issues:
            row = self.all_df.iloc[index]
            output = {
                'data': {
                    'text': str(row.review),
                },
                'predictions': [{
                    'result': [
                        {
                            'value': {
                                'choices': [
                                    'Positive' if row.label == 1 else 'Negative'
                                ]
                            },
                            'id': f'data-{index}',
                            'from_name': 'sentiment',
                            'to_name': 'text',
                            'type': 'choices',
                        },
                    ],
                }],
            }
            outputs.append(output)

        # save to file
        preanno_path = join(self.config.review.save_dir, 'pre-annotations.json')
        to_json(outputs, preanno_path)

        self.next(self.retrain_retest)

    @step
    def retrain_retest(self):
        r"""Retrain without reviewing. Let's assume all the labels that 
        confidence learning suggested to flip are indeed erroneous."""
        dm = ReviewDataModule(self.config)
        train_size = len(dm.train_dataset)
        dev_size = len(dm.dev_dataset)

        dm.train_dataset.data = self.all_df.iloc[:train_size]
        dm.dev_dataset.data = self.all_df.iloc[train_size:train_size+dev_size]
        dm.test_dataset.data = self.all_df.iloc[train_size+dev_size:]

        # start from scratch
        system = SentimentClassifierSystem(self.config)
        trainer = Trainer(max_epochs=self.config.train.optimizer.max_epochs)

        trainer.fit(system, dm)
        trainer.test(system, dm, ckpt_path='best')
        results = system.test_results

        pprint(results)

        log_file = join(LOG_DIR, 'conflearn.json')
        os.makedirs(LOG_DIR, exist_ok=True)
        to_json(results, log_file)  # save to disk

        self.next(self.end)

    @step
    def end(self):
        """End node!"""
        print('done! great work!')


if __name__ == "__main__":
    flow = TrainIdentifyReview()
