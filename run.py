# -*- coding:utf-8 -*-
import networkx as nx
from userinfluence import createGraph, smallworld, dimensionless, userActivity, PageRank, SocialRank, \
    createEvaluationGraph, calculateWeights, LTM, User

import random

graph = createGraph()
# smallworld(graph=graph)
# dimensionless(graph=graph)

start_time = "2014-01-01"
end_time = "2014-04-01"
activity = userActivity(start_time=start_time, end_time=end_time)

seednums = 30

print "开始PageRank算法..."
pagerank = PageRank(graph=graph)
pagerank.rank()
pagerank_chooseusers = pagerank.chooseUsers(user_num=seednums)
print "pagerank算法计算所得的用户：", pagerank_chooseusers

print "开始socialrank算法..."
socialrank = SocialRank(graph=graph, activity=activity)
socialrank.rank()
socialrank_chooseusers = socialrank.chooseUsers(user_num=seednums)
print "socialrank算法计算所得的用户：", socialrank_chooseusers

pagerank_users = User(users=pagerank_chooseusers, activity=activity)
socialrank_users = User(users=socialrank_chooseusers, activity=activity)

print "pagerank用户的活跃度：", pagerank_users.userActivity()
print "pagerank用户的粉丝数：", pagerank_users.followerscount()

print "socialrank用户的活跃度：", socialrank_users.userActivity()
print "socialrank用户的粉丝数：", socialrank_users.followerscount()

"""创建评估拓扑结构图,以及计算权重和线性阈值模型"""
evaluation_graph = createEvaluationGraph(activity=activity)
weights = calculateWeights(evaluation_graph)
ltm = LTM(evaluation_graph, weights)

"""首先检测模型是否符合条件"""
print "线性阈值模型是否满足条件：", ltm.checkLTM()

results = dict()
methods = ['socialrank', 'pagerank']
for method in methods:
    results[method] = dict().fromkeys([i+5 for i in range(0, seednums, 5)], 0)

iterator_num = 5
for i in range(iterator_num):
    thresholds = dict()  # threshold for each node
    for node in ltm.graph:
        thresholds[node] = random.random()

    for choosenum in range(0, len(socialrank_chooseusers), 5):
        activedusers = ltm.runLTM(initnodes=socialrank_chooseusers[: choosenum+5], thresholds=thresholds)
        results['socialrank'][choosenum+5] += len(activedusers)
        print "迭代次数：", i+1, ";socialrank选取用户数：", choosenum+5, ";激活的用户数：", len(activedusers)

    for choosenum in range(0, len(pagerank_chooseusers), 5):
        activedusers = ltm.runLTM(initnodes=pagerank_chooseusers[: choosenum+5], thresholds=thresholds)
        results['pagerank'][choosenum+5] += len(activedusers)
        print "迭代次数：", i+1, ";pagerank选取的用户数：", choosenum+5, ";激活的用户数：", len(activedusers)

for method in methods:
    for choosenum, activedusersnum in results[method].items():
        print "影响力方法：", method, ";选取的用户数：", choosenum, ";最终激活的用户数：", activedusersnum / iterator_num
