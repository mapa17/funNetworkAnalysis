import logging
import pandas as pd
import networkx as nx
import scipy as sp
import numpy as np
import traceback


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


def write_nodes_to_csv(graph, outputFile):
    try:
        node_data = graph.nodes(data=True)
        if node_data == []:
            logging.warning('Empty graph! Nothing to write to file!')
        else:
            names = [r[0] for r in node_data]
            columns_names = node_data[0][1].keys()
            data = {}
            for col in columns_names:
                data[col] = [r[1][col] for r in node_data]
            table = pd.DataFrame(index=names, data=data)
            if table is not None:
                table.to_csv(outputFile)
            return table
    except Exception as e:
        logging.error('Writing node data to csv failed! %s', traceback.format_exc())


def get_nodes_as_table(graph):
    """
    Generate a pandas table with nodes as index and their properties as columns
    """
    try:
        node_data = graph.nodes(data=True)
        if node_data == []:
            logging.warning('Empty graph! Nothing to write to file!')
            table = pd.DataFrame()
        else:
            names = [r[0] for r in node_data]
            columns_names = node_data[0][1].keys()
            data = {}
            for col in columns_names:
                data[col] = [r[1][col] for r in node_data]
            table = pd.DataFrame(index=names, data=data)
        return table
    except Exception as e:
        logging.error('Generating table for nodes failed! %s', traceback.format_exc())


def write_edges_to_csv(graph, outputFile):
    try:
        ed = graph.edges(data=True)
        if ed == []:
            logging.warning('Empty graph! Nothing to write to file!')
        else:
            index = [(e[0], e[1]) for e in ed]
            columns_names = ed[0][2].keys()
            data = {}
            for col in columns_names:
                data[col] = [e[2][col] for e in ed]
            table = pd.DataFrame(index=index, data=data)
            if outputFile is not None:
                table.to_csv(outputFile)
            return table
    except Exception as e:
        logging.error('Writing node data to csv failed! %s', traceback.format_exc())
