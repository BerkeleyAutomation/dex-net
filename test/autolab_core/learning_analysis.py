# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Helper classes for analyzing machine learning output
Author: Jeff Mahler
"""
import numpy as np
import os
import ruamel.yaml as yaml

import sklearn.metrics as sm
import scipy.stats as ss

class ConfusionMatrix(object):
    """ Confusion matrix for classification errors """
    def __init__(self, num_categories):
        self.num_categories = num_categories

        # organized as row = true, column = pred
        self.data = np.zeros([num_categories, num_categories])

    def update(self, predictions, labels):
        num_pred = predictions.shape[0]
        for i in range(num_pred):
            self.data[labels[i].astype(np.uint16), predictions[i].astype(np.uint16)] += 1

class ClassificationResult(object):
    def __init__(self, pred_probs, labels):
        """ Creates a classification result.

        Parameters
        ----------
        pred_probs : :obj:`numpy.ndarray`
            array of predicted class probabilities
        labels : :obj:`numpy.ndarray`
            array of integer class labels
        """
        if not isinstance(pred_probs, np.ndarray):
            pred_probs = np.array(pred_probs)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)        
        self.pred_probs = pred_probs.astype(np.float32)
        self.labels = labels.astype(np.uint32)

    @property
    def error_rate(self):
        return 100.0 - (
            100.0 *
            np.sum(self.predictions == self.labels) /
            self.num_datapoints)

    @property
    def accuracy(self):
        return np.mean(1 * (self.predictions == self.labels))

    def top_k_error_rate(self, k):
        predictions_arr = self.top_k_predictions(k)
        labels_arr = np.zeros(predictions_arr.shape)
        for i in range(k):
            labels_arr[:,i] = self.labels

        return 100.0 - (
            100.0 *
            np.sum(predictions_arr == labels_arr) /
            self.num_datapoints)

    @property
    def fpr(self):
        if np.sum(self.labels == 0) == 0:
            return 0.0
        return float(np.sum((self.predictions == 1) & (self.labels == 0))) / np.sum(self.labels == 0)

    @property
    def precision(self):
        if np.sum(self.predictions == 1) == 0:
            return 1.0
        return float(np.sum((self.predictions == 1) & (self.labels == 1))) / np.sum(self.predictions == 1)

    @property
    def recall(self):
        if np.sum(self.predictions == 1) == 0:
            return 1.0
        return float(np.sum((self.predictions == 1) & (self.labels == 1))) / np.sum(self.labels == 1)

    @property
    def num_datapoints(self):
        return self.pred_probs.shape[0]

    @property
    def num_categories(self):
        return self.pred_probs.shape[1]
        
    @property
    def predictions(self):
        return np.argmax(self.pred_probs, 1)

    def top_k_predictions(self, k):
        return np.argpartition(self.pred_probs, -k, axis=1)[:, -k:]

    @property
    def confusion_matrix(self):
        cm = ConfusionMatrix(self.num_categories)
        cm.update(self.predictions, self.labels)
        return cm

    def mispredicted_indices(self):
        return np.where(self.predictions != self.labels)[0]

    def correct_indices(self):
        return np.where(self.predictions == self.labels)[0]

    def convert_labels(self, mapping):
        new_num_categories = len(set(mapping.values()))
        new_probs = np.zeros([self.num_datapoints, new_num_categories])
        new_labels = np.zeros(self.num_datapoints)
        for i in range(self.num_datapoints):
            for j in range(self.num_categories):
                new_probs[i,mapping[j]] += self.pred_probs[i,j]
            new_labels[i] = mapping[self.labels[i]]
        return ClassificationResult([new_probs], [new_labels])

    @property
    def label_vectors(self):
        label_mat = np.zeros(self.pred_probs.shape)
        for i in range(self.num_datapoints):
            label_mat[i, self.labels[i]] = 1

        pred_probs_vec = self.pred_probs.ravel()
        labels_vec = label_mat.ravel()
        return pred_probs_vec, labels_vec

    def precision_recall_curve(self, plot=False, line_width=2, font_size=15, color='b', style='-', label='', marker=None):
        pred_probs_vec, labels_vec = self.label_vectors
        precision, recall, thresholds = sm.precision_recall_curve(labels_vec, pred_probs_vec)
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(recall, precision, linewidth=line_width, color=color, linestyle=style, label=label, marker=marker)
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.xlabel('Recall', fontsize=font_size)
            plt.ylabel('Precision', fontsize=font_size)
        return precision, recall, thresholds

    def roc_curve(self, plot=False, line_width=2, font_size=15, color='b', style='-', label=''):
        pred_probs_vec, labels_vec = self.label_vectors
        fpr, tpr, thresholds = sm.roc_curve(labels_vec, pred_probs_vec)

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(fpr, tpr, linewidth=line_width, color=color, linestyle=style, label=label)
            plt.xlabel('FPR', fontsize=font_size)
            plt.ylabel('TPR', fontsize=font_size)
        return fpr, tpr, thresholds

    @property
    def ap_score(self):
        pred_probs_vec, labels_vec = self.label_vectors
        ap = 0.0
        try:
            ap = sm.average_precision_score(labels_vec, pred_probs_vec)
        except:
            pass
        return ap

    @property
    def auc_score(self):
        pred_probs_vec, labels_vec = self.label_vectors
        auc = 0.0
        try:
            auc = sm.roc_auc_score(labels_vec, pred_probs_vec)
        except:
            pass
        return auc
            
    @property
    def pearson_correlation(self):
        pred_probs_vec, labels_vec = self.label_vectors
        pcorr_coef = np.corrcoef(labels_vec, pred_probs_vec)
        pearson_coef = pcorr_coef[0,1]
        return pearson_coef

    @property
    def spearman_correlation(self):
        pred_probs_vec, labels_vec = self.label_vectors
        spearman_coef, _ = ss.spearmanr(labels_vec, pred_probs_vec)
        return spearman_coef

    @property
    def spearman_pvalue(self):
        pred_probs_vec, labels_vec = self.label_vectors
        _, p_value = ss.spearmanr(labels_vec, pred_probs_vec)
        return p_value

    def save(self, filename):
        if not os.path.exists(filename):
            os.mkdir(filename)
        
        pred_filename = os.path.join(filename, 'predictions.npz')
        np.savez_compressed(pred_filename, self.pred_probs)

        labels_filename = os.path.join(filename, 'labels.npz')
        np.savez_compressed(labels_filename, self.labels)

    @staticmethod
    def load(filename):
        if not os.path.exists(filename):
            raise ValueError('File %s does not exists' %(filename))

        pred_filename = os.path.join(filename, 'predictions.npz')
        pred_probs = np.load(pred_filename)['arr_0']

        labels_filename = os.path.join(filename, 'labels.npz')
        labels = np.load(labels_filename)['arr_0']
        return ClassificationResult(pred_probs, labels)

    @staticmethod
    def make_summary_table(train_result, val_result, plot=True, save_dir=None, prepend="", save=False):
        """
        Makes a matplotlib table object with relevant data.
        Thanks to Lucas Manuelli for the contribution.

        Parameters
        ----------
        train_result: ClassificationResult
            result on train split

        val_result: ClassificationResult
            result on validation split

        save_dir: str
            path pointing to where to save results

        Returns
        -------
        dict
            dict with stored values, can be saved to a yaml file

        :obj:`matplotlibt.pyplot.fig`
            a figure containing the table
        """
        table_key_list = ['error_rate', 'recall_at_99_precision', 'average_precision', 'precision', 'recall']
        num_fields = len(table_key_list)
        
        import matplotlib.pyplot as plt
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        data = np.zeros([num_fields, 2])
        data_dict = dict()

        names = ['train', 'validation']
        for name, result in zip(names, [train_result, val_result]):
            data_dict[name] = {}
            data_dict[name]['error_rate'] = result.error_rate
            data_dict[name]['average_precision'] = result.ap_score * 100
            data_dict[name]['precision'] = result.precision * 100
            data_dict[name]['recall'] = result.recall * 100

            precision_array, recall_array, _ = result.precision_recall_curve()
            recall_at_99_precision = recall_array[np.argmax(precision_array > 0.99)] * 100  # to put it in percentage terms
            data_dict[name]['recall_at_99_precision'] = recall_at_99_precision
                        
            for i, key in enumerate(table_key_list):
                data_dict[name][key] = float("{0:.2f}".format(data_dict[name][key]))
                j = names.index(name)
                data[i, j] = data_dict[name][key]

        table = plt.table(cellText=data, rowLabels=table_key_list, colLabels=names)

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.15)
        
        if plot:
            plt.show()

        # save the results
        if save_dir is not None and save:
            fig_filename = os.path.join(save_dir, prepend + 'summary.png')
            yaml_filename = os.path.join(save_dir, prepend + 'summary.yaml')
            
            yaml.dump(data_dict, open(yaml_filename, 'w'), default_flow_style=False)
            fig.savefig(fig_filename, bbox_inches="tight")
            
        return data_dict, fig
    
class RegressionResult(object):
    def __init__(self, predictions, labels):
        """ Creates a classification result.
        NOTE: Does not yet work with multidimensional data.

        Parameters
        ----------
        predictions : :obj:`numpy.ndarray`
            array of predicted values
        labels : :obj:`numpy.ndarray`
            array of true values
        """
        self.predictions = predictions
        self.labels = labels
        
    @property
    def mse(self):
        return np.sum((self.predictions - self.labels)**2) / self.num_datapoints

    @property
    def num_datapoints(self):
        return self.predictions.shape[0]

    def save(self, filename):
        if not os.path.exists(filename):
            os.mkdir(filename)
        
        pred_filename = os.path.join(filename, 'predictions.npz')
        np.savez_compressed(pred_filename, self.predictions)

        labels_filename = os.path.join(filename, 'labels.npz')
        np.savez_compressed(labels_filename, self.labels)

    @staticmethod
    def load(filename):
        if not os.path.exists(filename):
            raise ValueError('File %s does not exists' %(filename))

        pred_filename = os.path.join(filename, 'predictions.npz')
        predictions = np.load(pred_filename)['arr_0']

        labels_filename = os.path.join(filename, 'labels.npz')
        labels = np.load(labels_filename)['arr_0']
        return RegressionResult(predictions, labels)

class BinaryClassificationResult(ClassificationResult):
    def __init__(self, pred_probs, labels, threshold=0.5):
        self.threshold = threshold
        ClassificationResult.__init__(self, pred_probs, labels)

    @property
    def num_categories(self):
        return 2

    @property
    def label_vectors(self):
        return self.pred_probs, self.labels

    def top_k_predictions(self, k):
        raise NotImplementedError()

    def top_k_error_rate(self, k):
        raise NotImplementedError()

    @property
    def predictions(self):
        return 1 * (self.pred_probs >= self.threshold)

    @property
    def precision(self):
        if sum(self.predictions) == 0:
            return 1.0
        return sm.precision_score(self.labels, self.predictions)

    @property
    def recall(self):
        return sm.recall_score(self.labels, self.predictions)

    @property
    def tpr(self):
        return self.recall

    @property
    def fpr(self):
        fp = np.sum((self.labels == 0) & (self.predictions == 1))
        an = np.sum(self.labels == 0)
        return float(fp) / an

    @property
    def f1_score(self):
        return sm.f1_score(self.labels, self.predictions)

    @property
    def phi_coef(self):
        return sm.matthews_corrcoef(self.labels, self.predictions)

    @property
    def true_positive_indices(self):
        return np.where((self.labels == 1) & (self.predictions == 1))[0]

    @property
    def false_positive_indices(self):
        return np.where((self.labels == 0) & (self.predictions == 1))[0]

    @property
    def true_negative_indices(self):
        return np.where((self.labels == 0) & (self.predictions == 0))[0]

    @property
    def false_negative_indices(self):
        return np.where((self.labels == 1) & (self.predictions == 0))[0]
    
    @property
    def num_true_pos(self):
        return np.sum(self.labels)

    @property
    def num_true_neg(self):
        return self.num_datapoints - self.num_true_pos

    @property
    def num_false_pos(self):
        return self.false_positive_indices.shape[0]

    @property
    def num_false_neg(self):
        return self.false_negative_indices.shape[0]
    
    @property
    def pct_true_pos(self):
        return np.mean(self.labels)

    @property
    def pct_true_neg(self):
        return 1.0 - self.pct_true_pos

    @property
    def pct_pred_pos(self):
        return np.mean(1 * (self.predictions == 1))

    @property
    def pct_pred_neg(self):
        return 1.0 - self.pct_pred_pos

    @property
    def cross_entropy_loss(self):
        probs = self.pred_probs.copy()
        probs[probs == 1.0] = 0.99999
        probs[probs == 0.0] = 0.00001
        return -np.mean(self.labels * np.log(probs) + (1.0 - self.labels) * np.log(1.0 - probs))
    
    @property
    def sorted_values(self):
        # sort by prob
        labels_and_probs = zip(self.labels, self.pred_probs)
        labels_and_probs.sort(key = lambda x: x[1])
        labels = [l[0] for l in labels_and_probs]
        probs = [l[1] for l in labels_and_probs]
        return labels, probs

    @property
    def app_score(self):
        """ Computes the area under the app curve. """
        # compute curve
        precisions, pct_pred_pos, taus = self.precision_pct_pred_pos_curve(interval=False)

        # compute area
        app = 0
        total = 0
        for k in range(len(precisions)-1):
            # read cur data
            cur_prec = precisions[k]
            cur_pp = pct_pred_pos[k]
            cur_tau = taus[k]

            # read next data
            next_prec = precisions[k+1]
            next_pp = pct_pred_pos[k+1]
            next_tau = taus[k+1]
            
            # approximate with rectangles
            mid_prec = (cur_prec + next_prec) / 2.0
            width_pp = np.abs(next_pp - cur_pp)
            app += mid_prec * width_pp
            total += width_pp

        return app

    def precision_recall_curve(self, plot=False, line_width=2, font_size=15, color='b', style='-', label='', marker=None):
        precision, recall, thresholds = sm.precision_recall_curve(self.labels, self.pred_probs)
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(recall, precision, linewidth=line_width, color=color, linestyle=style, label=label, marker=marker)
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.xlabel('Recall', fontsize=font_size)
            plt.ylabel('Precision', fontsize=font_size)
        return precision, recall, thresholds

    def roc_curve(self, plot=False, line_width=2, font_size=15, color='b', style='-', label=''):
        fpr, tpr, thresholds = sm.roc_curve(self.labels, self.pred_probs)

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(fpr, tpr, linewidth=line_width, color=color, linestyle=style, label=label)
            plt.xlabel('FPR', fontsize=font_size)
            plt.ylabel('TPR', fontsize=font_size)
        return fpr, tpr, thresholds
    
    def accuracy_curve(self, delta_tau=0.01):
        """ Computes the relationship between probability threshold
        and classification accuracy. """
        # compute thresholds based on the sorted probabilities
        orig_thresh = self.threshold
        sorted_labels, sorted_probs = self.sorted_values

        scores = []
        taus = []
        tau = 0
        for k in range(len(sorted_labels)):
            # compute new accuracy
            self.threshold = tau
            scores.append(self.accuracy)
            taus.append(tau)

            # update threshold
            tau = sorted_probs[k]

        # add last datapoint
        tau = 1.0
        self.threshold = tau
        scores.append(self.accuracy)
        taus.append(tau)

        self.threshold = orig_thresh
        return scores, taus

    def precision_curve(self, delta_tau=0.01):
        """ Computes the relationship between probability threshold
        and classification precision. """
        # compute thresholds based on the sorted probabilities
        orig_thresh = self.threshold
        sorted_labels, sorted_probs = self.sorted_values

        scores = []
        taus = []
        tau = 0
        for k in range(len(sorted_labels)):
            # compute new accuracy
            self.threshold = tau
            scores.append(self.precision)
            taus.append(tau)

            # update threshold
            tau = sorted_probs[k]

        # add last datapoint
        tau = 1.0
        self.threshold = tau
        scores.append(1.0)
        taus.append(tau)

        self.threshold = orig_thresh
        return scores, taus

    def recall_curve(self, delta_tau=0.01):
        """ Computes the relationship between probability threshold
        and classification precision. """
        # compute thresholds based on the sorted probabilities
        orig_thresh = self.threshold
        sorted_labels, sorted_probs = self.sorted_values

        scores = []
        taus = []
        tau = 0
        for k in range(len(sorted_labels)):
            # compute new accuracy
            self.threshold = tau
            scores.append(self.recall)
            taus.append(tau)

            # update threshold
            tau = sorted_probs[k]

        # add last datapoint
        tau = 1.0
        self.threshold = tau
        scores.append(1.0)
        taus.append(tau)

        self.threshold = orig_thresh
        return scores, taus

    def f1_curve(self, delta_tau=0.01):
        """ Computes the relationship between probability threshold
        and classification F1 score. """
        # compute thresholds based on the sorted probabilities
        orig_thresh = self.threshold
        sorted_labels, sorted_probs = self.sorted_values

        scores = []
        taus = []
        tau = 0
        for k in range(len(sorted_labels)):
            # compute new accuracy
            self.threshold = tau
            scores.append(self.f1_score)
            taus.append(tau)

            # update threshold
            tau = sorted_probs[k]

        # add last datapoint
        tau = 1.0
        self.threshold = tau
        scores.append(self.f1_score)
        taus.append(tau)

        self.threshold = orig_thresh
        return scores, taus

    def phi_coef_curve(self, delta_tau=0.01):
        """ Computes the relationship between probability threshold
        and classification phi coefficient. """
        # compute thresholds based on the sorted probabilities
        orig_thresh = self.threshold
        sorted_labels, sorted_probs = self.sorted_values

        scores = []
        taus = []
        tau = 0
        for k in range(len(sorted_labels)):
            # compute new accuracy
            self.threshold = tau
            scores.append(self.phi_coef)
            taus.append(tau)

            # update threshold
            tau = sorted_probs[k]

        # add last datapoint
        tau = 1.0
        self.threshold = tau
        scores.append(self.phi_coef)
        taus.append(tau)

        self.threshold = orig_thresh
        return scores, taus

    def precision_pct_pred_pos_curve(self, interval=False, delta_tau=0.001):
        """ Computes the relationship between precision
        and the percent of positively classified datapoints . """
        # compute thresholds based on the sorted probabilities
        orig_thresh = self.threshold
        sorted_labels, sorted_probs = self.sorted_values

        precisions = []
        pct_pred_pos = []
        taus = []
        tau = 0
        if not interval:
            for k in range(len(sorted_labels)):
                # compute new accuracy
                self.threshold = tau
                precisions.append(self.precision)
                pct_pred_pos.append(self.pct_pred_pos)
                taus.append(tau)
                
                # update threshold
                tau = sorted_probs[k]

        else:
            while tau < 1.0:
                # compute new accuracy
                self.threshold = tau
                precisions.append(self.precision)
                pct_pred_pos.append(self.pct_pred_pos)
                taus.append(tau)
                
                # update threshold
                tau += delta_tau

        # add last datapoint
        tau = 1.0
        self.threshold = tau
        precisions.append(self.precision)
        pct_pred_pos.append(self.pct_pred_pos)
        taus.append(tau)

        precisions.append(1.0)
        pct_pred_pos.append(0.0)
        taus.append(1.0 + 1e-12)

        self.threshold = orig_thresh
        return precisions, pct_pred_pos, taus

