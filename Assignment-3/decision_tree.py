import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features: List[List[float]], labels: List[int]):
        # init.
        assert(len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.max(labels)+1


        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features: List[List[float]]) -> List[int]:
        y_pred = []
        for feature in features:
            y_pred.append(self.root_node.predict(feature))
        return y_pred

    def print_tree(self, node=None, name='node 0', indent=''):
        if node is None:
            node = self.root_node
        print(name + '{')
        if node.splittable:
            print(indent + '  split by dim {:d}'.format(node.dim_split))
            for idx_child, child in enumerate(node.children):
                self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
        else:
            print(indent + '  cls', node.cls_max)
        print(indent+'}')


class TreeNode(object):
    def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls

        count_max = 0
        #print(self.labels)
        for label in np.asarray(np.unique(labels)):
            #print("label",label)
            #print("self.labels.count",self.labels.count(label))
            if self.labels.count(label) > count_max:
                #print("count",self.labels.count(label))
                count_max = labels.count(label)
                self.cls_max = label # majority of current node
        #print ("clf.max",self.cls_max)
        #print("len of unique labels",len(np.unique(labels)))
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None # the dim of feature to be splitted

        self.feature_uniq_split = [] # the feature to be splitted


    def split(self):
        def conditional_entropy(branches: List[List[int]]) -> float:
            '''
            branches: C x B array,
                      C is the number of classes,
                      B is the number of branches
                      it stores the number of
            '''
            ########################################################
            # TODO: compute the conditional entropy
            ########################################################
            branches=np.array(branches)
            #print(branches)
            total_sum=np.sum(branches)
            #print("total_sum",total_sum)
            branch_sum=np.sum(branches,axis=0)
            entropy=0.0
            for i in range(len(branches[0])):
                each_branch=branches[:,i]
                #print("each_branch",each_branch)
                each_branch_entropy=0
                for j in each_branch:
                    if(j/branch_sum[i]==0):
                        each_branch_entropy+=0
                    else:
                        each_branch_entropy+=((j/branch_sum[i])*np.log(j/branch_sum[i]))
                #print("each_branch_entropy",each_branch_entropy)
                entropy+=((branch_sum[i]/total_sum)*((each_branch_entropy)*-1))
                #print("entropy",entropy)
            #print("entropy",entropy)
            return entropy

        entropies=[]
        if(len(self.features[0])==0):
            self.splittable=False
            return

        for idx_dim in range(len(self.features[0])):
            #print(self.labels)
            ############################################################
            # TODO: compare each split using conditional entropy
            #       find the best split
            ############################################################
            features_a=np.array(self.features)
            labels_a=np.array(self.labels)
            unique_values=np.array(np.unique(features_a[:,idx_dim]))
            dicts=[]
            keys=[]
            for i in unique_values:
                x=labels_a[features_a[:,idx_dim]==i]
                unique, counts = np.unique(x, return_counts=True)
                temp_dict=dict(zip(unique, counts))
                unique_keys=temp_dict.keys()
                keys.append(list(unique_keys))
                dicts.append(temp_dict)

            keys_s=set()
            for i in keys:
                for j in i:
                    keys_s.add(j)
            #print(keys_s)
            mainlist=[]
            listy=[]
            for key in keys_s:
                listy=[]
                for tdict in dicts:

                    if key in tdict:
                        listy.append(tdict.get(key))
                    else:
                        listy.append(0)
                mainlist.append(listy)
            entropies.append(conditional_entropy(mainlist))
        ############################################################
        # TODO: split the node, add child nodes
        ############################################################
        self.dim_split=np.argmin(entropies)
        features_a=np.array(self.features)
        labels_a=np.array(self.labels)
        self.feature_uniq_split=np.array(np.unique(features_a[:,self.dim_split])).tolist()
        for i in self.feature_uniq_split:
             features_topass=features_a[features_a[:,self.dim_split]==i]
             features_topass=np.delete(features_topass,self.dim_split,1)
             labels_topass=labels_a[features_a[:,self.dim_split]==i]
             num_cls=len(np.unique(labels_topass))
             self.children.append(TreeNode(features_topass.tolist(),labels_topass.tolist(), num_cls))

        # split the child nodes
        for child in self.children:
            if child.splittable:
                child.split()

        return

    def predict(self, feature: List[int]) -> int:
        if self.splittable:
            idx_child = self.feature_uniq_split.index(feature[self.dim_split])
            feature = feature[:self.dim_split] + feature[self.dim_split+1:]
            return self.children[idx_child].predict(feature)
        else:
            return self.cls_max



