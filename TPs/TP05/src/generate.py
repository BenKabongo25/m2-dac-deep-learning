# AMAL Master DAC
# Novembre 2023

# Ben Kabongo
# M2 MVA

import math
import torch

from textloader import  *


def generate(model, start='', max_length=200):
    prediction = torch.tensor([lettre2id[start]])
    codes = []
    with torch.no_grad():
        h = torch.zeros(1, model.hidden_dim)
        for _ in range(max_length):
            h, output = model(prediction, h)
            prediction = torch.softmax(output.squeeze(), dim=1).argmax(1).squeeze()
            code = prediction.item()
            codes.append(code)
            if code == EOS_IX:
                break
    text = code2string(codes)
    return text


class BeamTree:
    """ Noeud pour l'arbre de recherche Beam Search """
    def __init__(self, code, prob):
        self.code = code            # code du noeud
        self.prob = prob            # log proba du noeud
        self.cum_child_prob = 0     # meilleure log proba cumulée des noeuds fils
        self.children = []          # noeuds fils
        self.best_child = None      # meilleur noeud fils
        self.predicted = False      # la prédiction a été faite pour ce noeud
        self.index = 0              # index de la prédiction dans la séquence


def find_best(tree):
    """ Recherche du chemin qui maximise la log probabilité"""
    if tree.children == []:
        return tree
    best_child = None
    best_cum_child_prob = 0
    for child in tree.children:
        child = find_best(child)
        cum_child_prob = child.prob + child.cum_child_prob
        if cum_child_prob > best_cum_child_prob:
            best_cum_child_prob = cum_child_prob
            best_child = child
    tree.best_child = best_child
    tree.cum_child_prob = best_cum_child_prob
    return tree


def _generate_node(model, K, node, max_length):
    """ Génération des K meilleures prédictions pour un noeud Beam Search """

    # on arrête si le noeud a un code de fin de phrase ou si l'index max est atteint
    if node.predicted or node.code == EOS_IX or node.index == max_length:
        node.predicted = True
        return node

    # on prédit et on retient les K meilleures prédictions
    prediction = torch.tensor([node.code])
    with torch.no_grad():
        h = torch.zeros(1, model.hidden_dim)
        h, output = model(prediction, h)
        log_probs, indices = torch.topk(output, K, dim=1)
        children = []
        for code, prob in zip(indices, log_probs):
            child = BeamTree(code, prob)
            child.index = node.index + 1
            child.predicted = False
            child = _generate_node(model, K, child, max_length)
            children.append(child)
        node.children = children

    return node


def _generate_node_nucleus(model, alpha, node, max_length):
    """ Génération avec du nucleus sampling pour un noeud Beam Search """

    # on arrête si le noeud a un code de fin de phrase ou si l'index max est atteint
    if node.predicted or node.code == EOS_IX or node.index == max_length:
        node.predicted = True
        return node

    # on prédit et on retient les meilleures prédictions
    prediction = torch.tensor([node.code])
    with torch.no_grad():
        h = torch.zeros(1, model.hidden_dim)
        h, output = model(prediction, h)

        sorted_logits, sorted_indices = torch.sort(output, descending=True, dim=-1)
        cum_probs = torch.cumsum(sorted_logits, dim=-1)

        idx = cum_probs <= alpha # indices à conserver
        idx[:, 0] = 1 # au moins 1 indice

        log_probs = torch.masked_select(output, idx)
        codes = torch.masked_select(torch.argsort(sorted_indices, dim=-1), idx)

        children = []
        for code, prob in zip(codes, log_probs):
            child = BeamTree(code, prob)
            child.index = node.index + 1
            child.predicted = False
            child = _generate_node_nucleus(model, alpha, child, max_length)
            children.append(child)
        node.children = children

    return node


def generate_beam(model, K, start='', max_length=200, nucleus=False, alpha=.95):
    """ Génération avec du Beam Search """

    root = [BeamTree(lettre2id[start], 0)]
    if not nucleus:
        root = _generate_node(model, K, root, max_length)
    else:
        root = _generate_node_nucleus(model, alpha, root, max_length)

    # génération la plus vraisemblable
    root = find_best(root)

    # codes de la meilleure génération
    codes = []
    root = root.best_child # commencer après start
    while root is not None:
        codes.append(root.code)
        root = root.best_child

    # décodage
    text = code2string(codes)
    return text

