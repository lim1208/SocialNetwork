# -*- coding:utf-8 -*-
import math
import pymysql
import networkx as nx
import pylab as pl
from copy import deepcopy

import time

host = "127.0.0.1"
port = 3306
username = "root"
password = "root"
database = "twitter"


def createGraph():
    """根据用户之间的社交关系构建社交网络拓扑结构"""
    connect = pymysql.connect(host=host, port=port, user=username, password=password,
                              database=database, charset="utf8", use_unicode=True)
    cursor = connect.cursor()
    relation_sql = "SELECT desid AS followerId, srcid AS userId FROM chinese_relation"
    cursor.execute(relation_sql)
    relations = cursor.fetchall()
    connect.commit()
    print "完成获取社交网络用户之间的社交关系，总共构建的关系数量为：", relations.__len__()

    graph = nx.DiGraph()
    for followerId, userId in relations:
        graph.add_edge(followerId, userId)
    print "完成根据关系构建的社交网络拓扑结构，节点数为：", graph.number_of_nodes(), "关系数为：", graph.number_of_edges()

    return graph


def smallworld(graph=None):
    """社交网络小世界现象分析"""
    if graph is None:
        print "请先构建社交网络网络拓扑结构"
    else:
        smallworld_distance = nx.average_shortest_path_length(graph)
        print "该社交网络拓扑结构，平均最短路径长度为：", smallworld_distance


def dimensionless(graph=None):
    """社交网络无标度特征分析"""
    if graph is None:
        print "请先构建社交网络网络拓扑结构"
    else:
        degreeDist = dict()
        for n, degree in graph.degree_iter():
            if degree in degreeDist:
                degreeDist[degree] += 1
            else:
                degreeDist[degree] = 1
        keys = list()
        values = list()
        for degree, usercount in degreeDist.items():
            keys.append(math.log(degree+1))
            values.append(math.log(usercount+1))
        pl.plot(keys, values, 'o')
        pl.xlabel("the number of user's degree")
        pl.ylabel("the number of user")
        pl.savefig("pic/degree-log-log")
        pl.show()


def userActivity(start_time, end_time):
    """计算用户的活跃度，在时间区间内用户发布信息的天数的比例"""
    activity = dict()  # 存储社交网络用户的活跃度

    start_day = time.mktime(time.strptime(start_time, '%Y-%m-%d'))
    end_day = time.mktime(time.strptime(end_time, '%Y-%m-%d'))
    days = (end_day - start_day) / (60*60*24) + 1
    print "评估用户活跃度的时间区间为：", days

    connect = pymysql.connect(host=host, port=port, user=username, password=password,
                              database=database, charset="utf8", use_unicode=True)
    cursor = connect.cursor()
    activity_sql = "SELECT userid, COUNT(*) AS activitydays FROM " \
                   "(SELECT userid FROM chinese_tweet WHERE createdate BETWEEN '"+ start_time +"' AND '"+end_time+\
                   "' GROUP BY userid, DATE_FORMAT(createdate, '%Y-%m-%d')) as tmp GROUP BY userid"
    cursor.execute(activity_sql)
    results = cursor.fetchall()
    connect.commit()

    for userid, activitydays in results:
        activity[userid] = math.sqrt(activitydays / float(days))

    return activity


class PageRank(object):
    """
        构建的网络拓扑结构中的关系表示的是关注关系
        例如A->B表示的是用户A关注了用户B,用户A是用户B的粉丝
    """
    def __init__(self, graph, d=0.85):
        self.graph = graph
        self.V = graph.number_of_nodes()
        self.d = d
        self.ranks = dict()

    def rank(self, tol=1.0e-10, iterator_num=200):
        for node in self.graph.nodes():
            self.ranks[node] = 1/float(self.V)

        weights = dict()
        for node in self.graph.nodes():
            followers = self.graph.predecessors(node)
            for follower in followers:
                weights[(follower, node)] = 1 / float(len(self.graph.successors(follower)))

        for iterator in range(iterator_num):
            print "PageRank iterator:", iterator+1
            last_rank = deepcopy(self.ranks)
            for node in self.graph.nodes():
                rank_sum = 0
                followers = self.graph.predecessors(node)  # 当前用户的所有粉丝用户
                for follower in followers:
                    rank_sum += weights[(follower, node)] * self.ranks[follower]
                self.ranks[node] = (1-float(self.d)) * (1/float(self.V)) + self.d * rank_sum
            error = sum([abs(self.ranks[user] - last_rank[user]) for user in self.graph])
            if error < len(self.graph) * tol:
                break
        print "完成PageRank算法。"

    def chooseUsers(self, user_num=10):
        """默认选择影响力排名前10的社交网络用户"""
        if self.ranks.__len__() == 0:
            pass
        else:
            users = list()
            sorted_users = sorted(self.ranks.items(), key=lambda d: d[1], reverse=True)
            for i in range(user_num):
                users.append(sorted_users[i][0])
            return users


class SocialRank(object):
    def __init__(self, graph, activity, d=0.85):
        self.graph = graph
        self.activity = activity
        self.V = len(self.graph)
        self.d = d
        self.ranks = dict()

    def rank(self, tol=1.0e-10, iteractor_num=200):
        for node in self.graph.nodes():
            self.ranks[node] = 1 / float(self.V)

        weights = dict()
        for node in self.graph.nodes():
            followers = self.graph.predecessors(node)
            for follower in followers:
                activity_sum = sum([self.activity[user] for user in self.graph.successors(follower)])
                weights[(follower, node)] = 1 / float(activity_sum)

        for iterator in range(iteractor_num):
            print "SocialRank iterator:", iterator+1
            last_rank = deepcopy(self.ranks)
            for node in self.graph.nodes():
                rank_sum = 0
                followers = self.graph.predecessors(node)
                for follower in followers:
                    """根据用户活跃度进行影响力分配"""
                    rank_sum += (self.activity[node] * weights[(follower, node)]) * self.ranks[follower]
                self.ranks[node] = (1 - float(self.d)) * (1 / float(self.V)) + self.d * rank_sum
            error = sum([abs(self.ranks[user] - last_rank[user]) for user in self.graph])
            if error < len(self.graph) * tol:
                break
        print "完成SocialRank算法。"

    def chooseUsers(self, user_num=10):
        """默认选择影响力排名前10的社交网络用户"""
        if self.ranks.__len__() == 0:
            pass
        else:
            users = list()
            sorted_users = sorted(self.ranks.items(), key=lambda d: d[1], reverse=True)
            for i in range(user_num):
                users.append(sorted_users[i][0])
            return users


def createEvaluationGraph(activity=None):
    """根据用户之间的社交关系构建社交网络拓扑结构"""
    connect = pymysql.connect(host=host, port=port, user=username, password=password,
                              database=database, charset="utf8", use_unicode=True)
    cursor = connect.cursor()
    relation_sql = "SELECT desid AS followerId, srcid AS userId FROM chinese_relation"
    cursor.execute(relation_sql)
    relations = cursor.fetchall()
    connect.commit()
    print "完成获取社交网络用户之间的社交关系，总共构建的关系数量为：", relations.__len__()

    graph = nx.DiGraph()
    if activity is None:
        for followerId, userId in relations:
            graph.add_edge(followerId, userId, weight=1.0)
    else:
        for followerId, userId in relations:
            graph.add_edge(followerId, userId, weight=activity[followerId])
    print "完成根据关系构建的社交网络拓扑结构，节点数为：", graph.number_of_nodes(), "关系数为：", graph.number_of_edges()

    return graph


def calculateWeights(graph):
    """计算有向权重图的边的权重之和"""
    weights = dict()
    for node in graph:
        in_edges = graph.in_edges(node, data=True)
        weightsum = sum(edata['weight'] for v1, v2, edata in in_edges)
        for v1, v2, _ in in_edges:
            weights[(v1, v2)] = 1 / float(weightsum)
    return weights


class LTM(object):
    def __init__(self, graph, weights):
        """
            线性阈值模型中图的边关系是用户影响其他用户，
            例如A->B表示的是用户A对用户B的影响，也就是用户B是用户A的粉丝
        """
        self.graph = graph
        self.weights = weights

    def checkLTM(self, eps=1e-4):
        """To verify the sum of all incoming weights<=1"""
        for node in self.graph:
            in_edges = self.graph.in_edges(node, data=True)
            total = 0
            for v1, v2, edata in in_edges:
                total += self.weights[(v1, v2)] * self.graph[v1][v2]['weight']
            if total >= 1 + eps:
                return False
        return True

    def runLTM(self, initnodes, thresholds):
        """根据初始的节点和节点的激活值计算最终激活的用户数数量"""
        finalnodes = deepcopy(initnodes)  # target nodes

        W = dict(zip(self.graph.nodes(), [0] * len(self.graph)))
        sj = deepcopy(initnodes)

        while len(sj):
            snew = list()  # 激活的用户
            for u in sj:
                for v in self.graph[u]:
                    if v not in finalnodes:
                        W[v] += self.weights[(u, v)] * self.graph[u][v]['weight']
                        if W[v] >= thresholds[v]:
                            snew.append(v)  # 新添加的用户
                            finalnodes.append(v)
            sj = deepcopy(snew)
        return finalnodes

    def avgLTM(self, initnodes, thresholds, iterator_num=50):
        activedUserSize = 0
        for i in range(iterator_num):
            activedusers = self.runLTM(initnodes, thresholds)
            activedUserSize += len(activedusers) / iterator_num
        return activedUserSize


class User(object):
    def __init__(self, users, activity):
        self.users = users  # 影响力用户
        self.activity = activity  # 用户的活跃度

    def followerscount(self):
        followers_result = dict()
        connect = pymysql.connect(host=host, port=port, user=username, password=password,
                                  database=database, charset="utf8", use_unicode=True)
        cursor = connect.cursor()
        followerscount_sql = "SELECT srcid AS userId, count(*) AS followerscount FROM chinese_relation WHERE srcid IN ("
        users_info = ""
        for i in range(len(self.users)):
            if i < len(self.users)-1:
                users_info += "'" + str(self.users[i]) + "', "
            else:
                users_info += "'" + str(self.users[i]) + "') "
        followerscount_sql += users_info + "GROUP BY srcid"
        cursor.execute(followerscount_sql)
        followers = cursor.fetchall()
        connect.commit()

        for userId, followerscount in followers:
            followers_result[userId] = followerscount

        return followers_result

    def userActivity(self):
        activity_result = dict()
        for user in self.users:
            activity_result[user] = self.activity[user]
        return activity_result
