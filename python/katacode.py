import collections
import time

loops = 0
class Sol:
    def threeSum(self, nums):
        global loops
        if len(nums) < 3:
            return []
        hashTab = collections.defaultdict(list)
        for i, num in enumerate(nums):
            loops+=1
            hashTab[num].append(i)

        finalSet = set()
        for ele in hashTab:
            target = -1*ele
            for ht, val in hashTab.items():
                loops+=1
                if ht == ele and len(val) < 2:
                    continue
                tmp = target - ht
                if tmp in hashTab:
                    if (tmp == ele or tmp == ht) and len(hashTab[tmp]) < 2:
                        continue
                    if (tmp == ele == ht) and len(hashTab[tmp]) < 3:
                        continue
                    srtArr = tuple(sorted([tmp, ht, ele]))
                    finalSet.add(srtArr)

        print("loops", loops)
        *finalSet, = finalSet
        return finalSet

    def search(self, arr, val):
        if not arr:
            return -1
        if len(arr) == 1:
            return arr.index(val) if (val == arr[0]) else -1

        midIdx = len(arr) // 2
        if arr[midIdx] > val:
            return self.search(arr[:midIdx], val)
        elif arr[midIdx] < val:
            return self.search(arr[midIdx:], val)
        else:
            return arr.index(val) if (val == arr[midIdx]) else -1

class Mix:
    def mix(self, s1, s2):

        from string import ascii_lowercase
        sDict = {ch:[0]*2 for ch in ascii_lowercase}
        for ch in filter(lambda x: x.islower(), s1):
            sDict[ch][0] += 1

        for ch in filter(lambda x: x.islower(), s2):
            sDict[ch][1] += 1

        dGet = lambda x, y: sDict[x][y]
        fltFunc = lambda x: dGet(x,0) > 1 or dGet(x,1) > 1
        finalDict = {}
        for ch in filter(fltFunc, sDict):
            val1 = dGet(ch,0)
            val2 = dGet(ch,1)
            if val1 > val2:
                chTop = '1:'  + ch*val1
                finalDict[chTop] = (-1*val1, 1, ch)
            elif val1 < val2:
                chTop = '2:' + ch*val2
                finalDict[chTop] = (-1*val2, 2, ch)
            else:
                chTop = '=:' + ch*val1
                finalDict[chTop] = (-1*val1, 3, ch)

        return '/'.join(sorted(finalDict, key=lambda x: finalDict[x]))

    def tests(self):
        from functools import partial
        myprint = partial(print, sep='\n')

    def _permutations(self, string):
        from itertools import permutations
        print(map(lambda x: ''.join(x), set(permutations(string))))

class IsIntersting:
    def is_interesting(self, number, awesome_phrases):
        """
        Any digit followed by all zeros: 100, 90000
        Every digit is the same number: 1111
        The digits are sequential, incementing†: 1234
        The digits are sequential, decrementing‡: 4321
        The digits are a palindrome: 1221 or 73837
        The digits match one of the values in the awesome_phrases array"""
        zeroTest = lambda x: not any([int(el) for el in x])
        sameDigits = lambda it: not list(filter(lambda x: x != it[0] ,it))
        seqVer = lambda seq, x, it: seq[x:x+len(it)] == it
        digitSeq = lambda seq, it: seqVer(seq, seq.index(it[0]), it)
        palindrom = lambda it, mid: (True if it[:mid] == it[mid:][::-1] else False) if (len(it) % 2) == 0 else (True if it[:mid] == it[mid+1:][::-1] else False)
        nbrInSeq = lambda x, it: x in it
        val = 0

        for st, nbr in filter(lambda x: x[1] > 99 , [(k, v) for k, v in zip( (2 ,1 , 1), range(number, number+3) ) ] ):
            sNbr = str(nbr)
            res = palindrom(sNbr, len(sNbr) // 2) or zeroTest(sNbr[1:]) or \
                sameDigits(sNbr) or digitSeq("1234567890", sNbr) or \
                digitSeq("9876543210", sNbr) or \
                nbrInSeq(nbr, awesome_phrases)

            if res: return st

        return val

    def tests(self):
        tests = [
            {'n': 1542, 'interesting': [1542, 0], 'expected': 0},
            {'n': 1336, 'interesting': [1337, 256], 'expected': 1},
            {'n': 1337, 'interesting': [1337, 256], 'expected': 2},
            {'n': 11208, 'interesting': [1337, 256], 'expected': 0},
            {'n': 11209, 'interesting': [1337, 256], 'expected': 1},
            {'n': 11211, 'interesting': [1337, 256], 'expected': 2},
        ]
        for t in tests:
            print (self.is_interesting(t['n'], t['interesting']), t['expected'])

class TimedClass:
    def timed(fn):
        from time import perf_counter
        from functools import wraps
        @wraps(fn)
        def inner(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            end = time.perf_counter()
            print("{0} took {1:.6f}s to run.".format(fn.__name__, end-start))
            return result

        return inner


    def _rec_fib(n):
        if n<=2:
            return 1
        return _rec_fib(n-1) + _rec_fib(n-2)

    @timed
    def rec_fib(n):
        result = _rec_fib(n)
        return result

    @timed
    def fib(n: int) -> int:
        """
        Calculate the fibanatcci number
        """
        fib1 = 1
        fib2 = 1
        for _ in range(3, n+1):
            fib1, fib2 = fib2, fib1+fib2

        return fib2

    def fib_reduce(n):
        from functools import reduce
        return reduce(lambda x, y: x-1 + x-2, [1,1] + [x for x in range(3, n+1)])

class PathFinder:
    """
    CodeWar path finder problem:
    https://www.codewars.com/kata/576986639772456f6f00030c/train/python
    """
    def path_finder(self, area):
        matrix = list(map(lambda x: [int(el) for el in x], area.split("\n")))
        N = len(matrix)
        movePat = ((0, 1), (0, -1), (1, 0), (-1, 0))

        pAdd = lambda x, y: (x[0]+y[0], x[1]+y[1])
        nodeIB = lambda x: (0 <= x[0] < N and 0 <= x[1] < N)
        mVal = lambda x: matrix[x[0]][x[1]]

        visitedNode = set()
        pathWeight = {}
        edTrack = {}
        pathQueue = []
        pathWeight[(0, 0)] = 0
        pathQueue.append((0, 0))
        while len(pathQueue) > 0:
            node = pathQueue.pop()
            visitedNode.add(node)
            childs =filter(lambda x: x not in visitedNode and nodeIB(x), [pAdd(x, node) for x in movePat])
            for child in childs:
                edgeWeight = pathWeight[node] + abs(mVal(node) - mVal(child))
                if child in pathWeight:
                    if pathWeight[child] > edgeWeight:
                        pathWeight[child] = edgeWeight
                else:
                    pathWeight[child] = edgeWeight
                edTrack[child] = pathWeight[child]

            if ( len(edTrack) > 0 ):
                minVal = min(edTrack, key=lambda x: edTrack[x])
                del edTrack[minVal]
                pathQueue.append(minVal)

        return pathWeight[(N-1, N-1)] if (N-1, N-1) in pathWeight else 0

    def __call__(self):
        return self.tests()

    def tests(self):
        """
        Tests for PathFinder
        """
        a = "\n".join([
        "000",
        "000",
        "000"
        ])

        b = "\n".join([
        "010",
        "010",
        "010"
        ])

        c = "\n".join([
        "010",
        "101",
        "010"
        ])

        d = "\n".join([
        "0707",
        "7070",
        "0707",
        "7070"
        ])

        e = "\n".join([
        "700000",
        "077770",
        "077770",
        "077770",
        "077770",
        "000007"
        ])

        f = "\n".join([
        "777000",
        "007000",
        "007000",
        "007000",
        "007000",
        "007777"
        ])

        g = "\n".join([
        "000000",
        "000000",
        "000000",
        "000010",
        "000109",
        "001010"
        ])
        # print(self.path_finder(a), 0)
        print("result: {0}, expected: {1}".format(self.path_finder(a), 0))
        print("result: {0}, expected: {1}".format(self.path_finder(b), 2))
        print("result: {0}, expected: {1}".format(self.path_finder(c), 4))
        print("result: {0}, expected: {1}".format(self.path_finder(d), 42))
        print("result: {0}, expected: {1}".format(self.path_finder(e), 14))
        print("result: {0}, expected: {1}".format(self.path_finder(f), 0))
        print("result: {0}, expected: {1}".format(self.path_finder(g), 4))
        # print(self.path_finder(c), 4)
        # print(self.path_finder(d), 42)
        # print(self.path_finder(e), 14)
        # print(self.path_finder(f), 0)
        # print(self.path_finder(g), 4)

class NumsIslands:
    """
    leetecode problems:
    """

    def numIslands2(self, m: int, n: int, positions: "List[List[int]]") -> "List[int]":
        """
        https://leetcode.com/problems/number-of-islands-ii/
        """
        if not (m and n):
            return 0

        from collections import defaultdict
        grid = [[0 for _ in range(n)] for _ in range(m)]
        islands = defaultdict(set)
        islandCounter = 0

        def BFS(node):
            pat = ((0,1), (0,-1), (1,0), (-1,0))
            inb = lambda x: 0 <= x[0] < m and 0 <= x[1] < n
            nonlocal islandCounter
            childs = filter(lambda x: inb(x) and grid[x[0]][x[1]],
                            map(lambda x: (x[0]+node[0], x[1]+node[1]),
                            pat))

            islandvisited = set()
            hasChilds = False
            for child in childs:
                hasChilds = True
                for k, island in islands.items():
                    if child in island:
                        islands[k].add(node)
                        islandvisited.add(k)

            if not hasChilds:
                islandCounter+=1
                islands[islandCounter].add(node)


            if len(islandvisited) > 1:
                islandCounter+=1
                islands[islandCounter] = set().union(*[islands[k] for k in islandvisited])
                for k in islandvisited:
                    del islands[k]

            return len(islands)

        finalarr = []
        for pos in positions:
            if not grid[pos[0]][pos[1]]:
                grid[pos[0]][pos[1]] = 1
                finalarr.append(BFS(tuple(pos)))
            else:
                finalarr.append(len(islands))

        # print("finalarr: ", finalarr)
        return finalarr



    def numIslands1(self, grid: "List[List[str]]") -> int:
        """
        https://leetcode.com/problems/number-of-islands/
        """

        islands = 0
        vNds = set()

        def BFS(tnode):
            nonlocal vNds
            vNds.add(tnode)
            in_bound = lambda x: 0 <= x[0] < len(grid) and 0 <= x[1] < len(grid[0])
            mvPat = ((0,1), (0,-1), (1, 0), (-1,0))
            pathQueue = [tnode]
            while len(pathQueue) > 0:
                node = pathQueue.pop()
                childs = filter(lambda x: in_bound(x) and grid[x[0]][x[1]] == '1' and not (x in vNds), map(lambda x: (x[0]+node[0], x[1]+node[1]) ,mvPat))
                for child in childs:
                    vNds.add(child)
                    pathQueue.append(child)

        for i, row in enumerate(grid):
            for j, col in enumerate(row):
                if col == "1" and (i, j) not in vNds:
                    islands+=1
                    BFS((i, j))

        return islands

    def __call__(self):
        return self.tests()

    def tests(self):
        return self.numIslands2(3,3,[[0,0],[0,1],[1,2],[2,1],[1,0],[0,0],[2,2],[1,2],[1,1],[0,1]])
        # return self.numIslands1([
        #     ["1","1","1","1","0"],
        #     ["1","1","0","1","0"],
        #     ["1","1","0","0","1"],
        #     ["0","0","1","0","0"]])

class Trie:
    """
    https://leetcode.com/problems/implement-trie-prefix-tree/
    """
    class TrieNode:
        def __init__(self):
            self.children = [0]*26 # Represent the number of alphanums
            self.is_word = False

    def __init__(self, *args, **kwargs):
        """
        Initialize your data structure here.
        """
        self.root = self.TrieNode()
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.test(*self.args, **self.kwargs)

    def _char_to_index(self,char):
        return ord(char) - ord("a")

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        start = self.root
        for char in word:
            index = self._char_to_index(char)
            if not start.children[index]:
                start.children[index] = self.TrieNode()
            start = start.children[index]
        start.is_word = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        start = self.root
        for char in word:
            index = self._char_to_index(char)
            if not start.children[index]:
                return False
            start = start.children[index]
        if start.is_word: return True
        return False

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        start = self.root
        for char in prefix:
            index = self._char_to_index(char)
            if not start.children[index]:
                return False
            start = start.children[index]
        return True 

    def test(self, elements, search, prefix):
        for el in elements:
            self.insert(el)

        res = []
        res.append(self.search(search)) 
        res.append(self.startsWith(prefix))
        res = *res, 
        return res

def testCalss(className, *args, **kwargs):
    obj = className(*args, **kwargs)
    print(className.__name__, obj())

testCalss(NumsIslands)