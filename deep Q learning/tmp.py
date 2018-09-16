import win_unicode_console
win_unicode_console.enable()

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class TreeNode:
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None

class Solution:
    def lastRemaining(self, n):
        """
        :type n: int
        :rtype: int
        """
        saver = []
        tmp = (1,n)
        flag = True    # True for left, and False for right 
        while tmp != (0,0) and tmp != (1,1):
            if flag == True:
                if (tmp[1]-tmp[0]+1)%2 == 0:
                    if tmp[0] == 1:
                        tmp = (1,int(tmp[1]/2))
                        saver.append(0)
                    else:
                        tmp = (0,int(tmp[1]/2))
                        saver.append(1)
                else:
                    if tmp[0] == 1:
                        tmp = (1,int(tmp[1]/2))
                        saver.append(0)
                    else:
                        tmp = (0,int((tmp[1]-1)/2))
                        saver.append(1)
                flag = False
            else:
                if (tmp[1]-tmp[0]+1)%2 == 0:
                    if tmp[0] == 0:
                        saver.append(0)
                    else:
                        saver.append(1)
                    tmp = (0,int((tmp[1]-1)/2))
                else:
                    if tmp[0] == 0:
                        tmp = (0,int((tmp[1]-1)/2))
                        saver.append(1)
                    else:
                        tmp = (1,int((tmp[1]-1)/2))
                        saver.append(0)
                flag = True
        res = tmp[0]
        for i in reversed(saver):
            res = res*2+i
        return res
                
                


class FenwickTree(object):
    def __init__(self, n):
        self.n = n
        self.sums = [0] * (n + 1)

    def add(self, x, val):
        while x <= self.n:
            self.sums[x] += val
            x += self.lowbit(x)

    def lowbit(self, x):
        return x & -x

    def sum(self, x):
        res = 0
        while x > 0:
            res += self.sums[x]
            x -= self.lowbit(x)
        return res


a = TreeNode(1)
b = TreeNode(2)
c = TreeNode(3)
d = TreeNode(4)
e = TreeNode(5)
f = TreeNode(6)
a.left = b
a.right = e
b.left = c
"""
c.next = d
d.next = e
e.next = f
"""
a1 = ListNode(3)
a2 = ListNode(4)
a3 = ListNode(1)
a4 = ListNode(4)
a5 = ListNode(5)

a1.next = a2
a2.next = a3

T = Solution()
res = T.lastRemaining(n = 9)
print(res)
