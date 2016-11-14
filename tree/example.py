#!/usr/bin/env python
# encoding: utf-8


import trees
import tree_plotter


fr = open('../materials/Ch03/lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lenses_labels = ['age', 'prescript', 'astigmatic', 'tear_rate']
lenses_tree = trees.create_tree(lenses, lenses_labels)
print(lenses_tree)
tree_plotter.create_plot(lenses_tree)
