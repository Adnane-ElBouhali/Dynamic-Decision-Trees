from typing import List, Tuple

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
        split_feature_index : int
            along which feature the points have been split
        split_value: float
            the value of the feature along which the points have been split
    
    """
    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes]):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """
        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.split_feature_index = None 
        self.split_value = None
    
    def get_gini(self) -> float:
        """Computes the Gini score of the set of points
        Returns
        -------
        float
            The Gini score of the set of points
        """
        n = len(self.features)
        n0 = sum(self.labels)
        n1 = n - n0
        gini = 1 - (n0/n)**2 - (n1/n)**2
        return gini
        

    def get_best_gain(self, min_split_points: int = 1) -> Tuple[int, float]:
        """Compute the feature along which splitting provides the best gain

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        """
        max_gini = 0.0
        index = None
        value = None
        gini = self.get_gini()

        for i, T in enumerate(self.types):
            unique_values = {feature[i] for feature in self.features}
            if T == FeaturesTypes.BOOLEAN:
                split_values = [None]
            elif T == FeaturesTypes.REAL:
                sorted_values = sorted(unique_values)
                split_values = [(sorted_values[i] + sorted_values[i + 1]) / 2 for i in range(len(sorted_values) - 1)]
            else:
                split_values = unique_values

            for k in split_values:
                gini_split = self.calculate_gini_split(i, k, min_split_points)
                if gini_split is not None:
                    gini_gain = gini - gini_split
                    if gini_gain > max_gini:
                        max_gini = gini_gain
                        index = i
                        value = k

        if max_gini == 0.0:
            return (None, None)

        self.split_feature_index = index
        self.split_value = value
        return (index, max_gini)    
    
    def calculate_gini_split(self, feature_index: int, split_value: float = None, min_split_points: int = 1) -> float:
        """
        Calculates the gini split

        Returns
        -------
        float
        The gini split
        
        """
        feature_type = self.types[feature_index]

        L1, L2 = [], []

        for i, T in enumerate(self.features):
            if feature_type == FeaturesTypes.BOOLEAN:
                test = T[feature_index]
            elif feature_type == FeaturesTypes.CLASSES:
                test = T[feature_index] == split_value
            else: 
                test = T[feature_index] < split_value
            
            (L1 if test else L2).append(i)

        if len(L1) == 0 or len(L2) == 0 or len(L1) < min_split_points or len(L2) < min_split_points:
            return None

        n01 = sum(1 for k in L1 if self.labels[k])
        n02 = len(L1) - n01
        n01 = n01 / len(L1)
        n02 = n02 / len(L1)

        n11 = sum(1 for k in L2 if self.labels[k])
        n12 = len(L2) - n11
        n11 = n11 / len(L2)
        n12 = n12 / len(L2)

        g1 = 1 - n01 ** 2 - n02 ** 2
        g2 = 1 - n11 ** 2 - n12 ** 2

        n = len(self.features)
        gini_split = (len(L1) / n) * g1 + (len(L2) / n) * g2
        return gini_split
       
    def get_best_threshold(self) -> float:
        """Calculates the optimal threshold value for the feature offering the highest gain.

            Returns
            -------
            float
    
        """

        if self.split_feature_index == None:
            return
        return self.split_value     

    def best_split(self, min_split_points: int = 1) -> Tuple[List[List[float]], List[bool], List[List[float]], List[bool]]:
        """Splits the training points based on the feature providing the best gini gain.

            Returns
            -------
            Tuple[List[List[float]], List[bool], List[List[float]], List[bool]]
    
        """
        self.get_best_gain(min_split_points)

        if self.split_feature_index is None:
            return (None, None, None, None)

        L01, L02 = [], []
        L11, L12 = [], []
        
        n = len(self.labels)
        feature_type = self.types[self.split_feature_index]
        split_value = self.split_value

        for i in range(n):
            feature_value = self.features[i][self.split_feature_index]

            condition = ((feature_type == FeaturesTypes.BOOLEAN and feature_value) or
                            (feature_type == FeaturesTypes.CLASSES and feature_value == split_value) or
                            (feature_type == FeaturesTypes.REAL and feature_value < split_value))

            if condition:
                L01.append(self.features[i])
                L02.append(self.labels[i])
            else:
                L11.append(self.features[i])
                L12.append(self.labels[i])

        return L01, L02, L11, L12
