import argparse
from os import error
import scipy
import joblib

import pandas as pd

from sklearn import ensemble
import sklearn
import numpy as np
from matplotlib import pyplot as plt
from lib.preprocessing import ChooseFeatureColumns
from lib.preprocessing import MyMapper
import logging

class TrainModel:
    def __init__(self, filename : str = None, runtime_label : str = "runtime"):
        # load the input file into a pandas dataframe
        try:
            df = pd.read_csv(filename)
        except error as e:
            df = pd.read_csv(filename, sep="\t")
        except:
            raise ValueError(f"unrecognized filetype: {filename}. I only accept tsv or csv files")
        
        # Pops the runtime_label into labels
        self.df_labels = df.pop(runtime_label)
        # Rest of the dataframe (minus label) is the features
        self.df_features = df

        # Return the natural logarithm of one plus the input array
        self.df_labels = np.log1p(self.df_labels)
        del df

    def runTraining(self, split_train_test=True, split_randomly=True):
        # prepare the data for the RandomForestRegressor
        logging.info("setting up...")

        # Create regressor and the pipeline
        # regr = ensemble.RandomForestRegressor(n_estimators=100, max_depth=12)

        regr = ensemble.RandomForestRegressor(n_estimators=500, max_depth=12, n_jobs=10)
        # regr = ensemble.AdaBoostRegressor()
        # regr = sklearn.linear_model.LinearRegression()
        self.pipeLine = sklearn.pipeline.Pipeline([
            ('chooser', ChooseFeatureColumns()),
            ('scaler', MyMapper()),
            ('regr', regr)
        ])

        # Split test/train 20%/80%
        test_size = 0.2
        test_start=len(self.df_labels)-int(len(self.df_labels)*test_size)
        logging.info(f"{test_start} {len(self.df_labels)}")

        # logging.info("self.args.split_randomly ", self.args.split_randomly)

        # Setup test/train split
        if split_train_test and split_randomly:
            tr_features, ev_features, tr_labels, ev_labels = sklearn.model_selection.train_test_split(self.df_features, self.df_labels, test_size=test_size)
            logging.info("splitting randomly")
        elif split_train_test:
            tr_features, tr_labels, ev_features, ev_labels = self.df_features[:test_start], self.df_labels[:test_start], self.df_features[test_start:], self.df_labels[test_start:]
            logging.info("splitting non-randomly")
        else:
            tr_features, tr_labels, ev_features, ev_labels = self.df_features, self.df_labels, self.df_features, self.df_labels
            logging.info("not splitting")


        logging.info("fitting the model...")
        self.pipeLine.fit(tr_features, tr_labels)
        ev_pred = self.pipeLine.predict(ev_features)

        # unlog the runtimes (they were previously log transformed in the function clean_data())
        self.cq = pd.DataFrame()
        self.cq["labels"] = np.expm1(ev_labels)
        self.cq["pred"] = np.expm1(ev_pred)

        # do the final analysis
        self.analysis_of_results()

        logging.info("done")

    def analysis_of_results(self):
        # some metrics
        r2_score = sklearn.metrics.r2_score(self.cq["labels"], self.cq["pred"])
        pearson = scipy.stats.pearsonr(self.cq["labels"], self.cq["pred"])
        mse = sklearn.metrics.mean_squared_error(self.cq["labels"], self.cq["pred"])

        logging.info(f"r2 score: {r2_score}")
        logging.info(f"pearson: {pearson[0]}")
        logging.info(f"mse: {mse}")

        # save model
        # joblib.dump(self.pipe, self.args.model_outfile)
        # logging.info(f"saved model to: {self.args.model_outfile}")

    def plot(self, plot_outfile : str = "plot.png"):
        plt.figure(figsize=(10,10))
        plt.scatter(self.cq["labels"], self.cq["pred"])
        plt.xlabel("Real Runtime")
        plt.ylabel("Predicted Runtime")
        plt.title("Mean predictions")
        plt.savefig(plot_outfile)
        logging.info(f"saved a plot to: {plot_outfile}")


def main():
    parser = argparse.ArgumentParser(description='Get the impact of tool features on it\'s runtime.',
                                     epilog='Accepts tsv and csv files')
    parser.add_argument('--filename', dest='filename', action='store', required=True)
    parser.add_argument('--debug', dest='debug', action='store_true', default=False)
    parser.add_argument("--runtime_label", dest='runtime_label', default="runtime")
    parser.add_argument("--split_train_test", dest='split_train_test', action='store_true', default="False")
    parser.add_argument("--split_randomly", dest='split_randomly', action='store_false', default="True")
    parser.add_argument('--plot_outfile', dest='plot_outfile', default="plot.png", help='png output file.')
    parser.add_argument('--model_outfile', dest='model_outfile', default='model.pkl', help='pkl output file.')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("pandas").setLevel(logging.WARNING)
        logging.getLogger("sklearn_pandas").setLevel(logging.WARNING)
        logging.basicConfig(level=logging.DEBUG)

    trainer = TrainModel(filename=args.filename)
    trainer.runTraining()
    trainer.plot()

    return 

if __name__ == '__main__':
    main()
    exit(0)