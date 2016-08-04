import jsonrpc
from simplejson import loads
server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
                             jsonrpc.TransportTcpIp(addr=("127.0.0.1", 8080)))


def getSubtreeFeatures(text):
    result = loads(server.parse(text))
    sents = result['sentences']
    numSents = len(sents)

    if numSents == 0:
        return (0,0,0,0)

    totalMaxD = 0
    totalMaxSub = 0
    maxDepth = 0
    maxSubTrees = 0
    for s in sents:
        ptree = s['parsetree']
        curDepth = 0
        curMaxD = 0
        numSubs = 0
        for c in ptree:
            if c == '(':
                curDepth += 1
                numSubs += 1
                curMaxD = max(curMaxD, curDepth)
            elif c == ')':
                curDepth -= 1
        maxSubTrees = max(maxSubTrees, numSubs)
        maxDepth = max(curMaxD, maxDepth)
        totalMaxD += curMaxD
        totalMaxSub += numSubs
    totalMaxD = totalMaxD/numSents
    totalMaxSub = totalMaxSub/numSents
    return (maxDepth, maxSubTrees, totalMaxD, totalMaxSub)
