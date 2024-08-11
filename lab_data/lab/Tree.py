from typing import List

from PointSet import PointSet, FeaturesTypes
    
class Node:
    """ Node of a decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the node
        left_node : Node
            The left node of the tree
        right_node : Node
            The right node of the tree
    """
    def __init__(self,
                 points: PointSet,
                 left_node: 'Node' = None,
                 right_node: 'Node' = None):
        """
        Parameters
        ----------
            points : PointSet
                The training points of the node
            left_node : Node
                The left node of the tree (split_feature = true)
            right_node : Node
                The right node of the tree (split_feature = false)
        """
        self.points = points
        self.left_node = left_node
        self.right_node = right_node
     
     
class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
        root : Node
            The root of the tree
        split_feature_index : int
            The index of the feature along which the points have been split
        split_value : float
            The value of the feature along which the points have been split
            for categorical feature, split_value(left child), the other value(right child)
            for real feature, less than split_value(left child), greater than split_value(right child)
            for boolean feature, split_value = None
    """
            
    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 h: int = 1,
                 min_split_points : int = 1):
        """
        Parameters
        ----------
            labels : List[bool]
                The labels of the training points.
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            types : List[FeaturesTypes]
                The types of the features.
            h : int default=1
                The height of the tree. The tree will have a maximum
                depth of leaf (the root is at depth 0).
        """

        self.points = PointSet(features,labels,types)
        self.root = Node(self.points,None,None)
        self.create(self.root,h,types,min_split_points)
        
        
    def create(self, node: Node, h: int, types, min_split_points: int = 1):
        """Creates a tree structure starting from the specified node.
                Parameters
                ----------
                h : int (The height of the given node, defined as the total height of the tree minus the depth of this node)
                types : List[FeaturesTypes] (A list indicating the types of features used in the tree)
        """

        if node.points.get_gini() == 0 or h == 0:
            return

        left_features, left_labels, right_features, right_labels = node.points.best_split(min_split_points)
        if node.points.split_feature_index is None:
            return

        node.left_node = Node(PointSet(left_features, left_labels, types), None, None)
        node.right_node = Node(PointSet(right_features, right_labels, types), None, None)

        self.create(node.left_node, h - 1, types, min_split_points)
        self.create(node.right_node, h - 1, types, min_split_points)


    def decide(self, features: List[float]) -> bool:
        """
        Given a set of features, return the class (True or False) in which the algorithm
        would put a point with those features.

        Return type
        -----------
        Bool
        """
        current_node = self.root

        while not (current_node.left_node is None and current_node.right_node is None):
            split_feature_index = current_node.points.split_feature_index
            split_value = current_node.points.split_value
            feature_type = current_node.points.types[split_feature_index]

            if ((feature_type == FeaturesTypes.BOOLEAN and features[split_feature_index]) or
                (feature_type == FeaturesTypes.CLASSES and features[split_feature_index] == split_value) or
                (feature_type == FeaturesTypes.REAL and features[split_feature_index] < split_value)):
                current_node = current_node.left_node
            else:
                current_node = current_node.right_node

        n1 = sum(current_node.points.labels)
        n2 = len(current_node.points.labels) - n1

        return n1 > n2
