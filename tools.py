import logging
import pandas as pd
import networkx as nx
import scipy as sp
import numpy as np

def replaceInColumn(T, nM, column):
    """
    Helper function that renames values from column in table, using nameMapping
    """
    try:
        srcNames = nM.ix[:, 0].tolist()
        destNames = nM.ix[:, 1].tolist()
        T.ix[:, column].replace(srcNames, destNames, inplace=True)
        return T
    except Exception as e:
        logging.error('Changing connectum to new names failed! %s', e)

def renameColumns(T, nM):
    """
    Helper function that renames column in table, using nameMapping
    """
    try:
        d = dict(zip(nM.iloc[:, 0].tolist(), nM.iloc[:, 1].tolist()))
        T.rename_axis(d, axis=1, inplace=True)
        C = T.columns.unique()
        for cn in C:
            part = T[cn]
            if isinstance(part, pd.DataFrame):
                reduced = part.mean(axis=1)
                T.drop(cn, axis=1, inplace=True)
                T[cn] = reduced
        return T
    except Exception as e:
        logging.error('Changing connectum to new names failed! %s', e)

def addEdges(G, row, threshold=0.0):
    destNode = row.name
    for srcNode, conn in row.iteritems():
        if not np.isnan(conn):
            if conn >= threshold:
                G.add_edge(srcNode, destNode, weight=conn)

def updateEdges(G, values, attrName):
    r = values.index.tolist()
    c = values.columns.tolist()
    nG = G.copy()
    for srcN, destN, attr in G.edges(data=True):
        if (srcN in r) and (destN in c):
            attr[attrName] = values.ix[srcN, destN]
            nG.add_edge(srcN, destN, **attr)

    return nG

def updateNodes(G, values, attrName):
    nG = G.copy()
    for node, attr in G.nodes(data=True):
        attr[attrName] = values[node]
        nG.add_node(node, **attr)
    return nG
