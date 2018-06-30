## 117 Populating Next Right Pointers in Each Node II

Follow up for problem "Populating Next Right Pointers in Each Node".

What if the given tree could be any binary tree? Would your previous solution still work?

Note:

You may only use constant extra space.
For example,
Given the following binary tree,

```
         1
       /  \
      2    3
     / \    \
    4   5    7
```
After calling your function, the tree should look like:
```
         1 -> NULL
       /  \
      2 -> 3 -> NULL
     / \    \
    4-> 5 -> 7 -> NULL
```
```javascript
/**
 * Definition for binary tree with next pointer.
 * function TreeLinkNode(val) {
 *     this.val = val;
 *     this.left = this.right = this.next = null;
 * }
 */

/**
 * @param {TreeLinkNode} root
 * @return {void} Do not return anything, modify tree in-place instead.
 */

var connect = function(root) {

    if (!root) return;

    let queue = [root];

    while(queue.length) {

        let level = [];
        let size = queue.length;

        for (let i = 0; i < size; i++){
            let currentNode = queue.shift();
            level.push(currentNode)
            if (currentNode.left) queue.push(currentNode.left);
            if (currentNode.right) queue.push(currentNode.right);
        }

        for (let i = 0; i < level.length - 1; i++) {
            if (level[i]) level[i].next = level[i + 1] ? level[i + 1] : null;
        }
    }

};
```
Use a breadth-first traversal to gather all the nodes on one level in an array called level. The size of the queue at each iteration of the while loop corresponds to the number of nodes on the current level of the traversal.

## 160 Intersection of Two Linked Lists
Write a program to find the node at which the intersection of two singly linked lists begins.

For example, the following two linked lists:

```
A:          a1 → a2
                   ↘
                     c1 → c2 → c3
                   ↗
B:     b1 → b2 → b3
```
begin to intersect at node c1.

Notes:

If the two linked lists have no intersection at all, return `null`.
The linked lists must retain their original structure after the function returns.
You may assume there are no cycles anywhere in the entire linked structure.
Your code should preferably run in `O(n)` time and use only `O(1)` memory.

```javascript
/**
 * Definition for singly-linked list.
 * function ListNode(val) {
 *     this.val = val;
 *     this.next = null;
 * }
 */

/**
 * @param {ListNode} headA
 * @param {ListNode} headB
 * @return {ListNode}
 */
var getIntersectionNode = function(headA, headB) {

    let lengthA = 0;
    let lengthB = 0;
    let currentNode = headA;

    while (currentNode) {
        lengthA++;
        currentNode = currentNode.next;
    }

    currentNode = headB;

    while (currentNode) {
        lengthB++;
        currentNode = currentNode.next;
    }

    let larger = lengthA > lengthB ? headA : headB;
    let smaller = lengthA > lengthB ? headB : headA;

    let difference = Math.abs(lengthA - lengthB);

    let currentLarger = larger;

    while (difference) {
        currentLarger = currentLarger.next;
        difference--;
    }

    let currentSmaller = smaller;

    while(currentLarger && currentSmaller) {
        if (currentLarger === currentSmaller) return currentLarger;

        currentLarger = currentLarger.next;
        currentSmaller = currentSmaller.next;

    }

    return null;
};
```

Get the difference between the lengths of the linked lists. Iterate through the larger list by the difference. Compare nodes with the identity operator to find the intersection. O(n) times and 0(1) space.

## 122 Best Time to Buy and Sell Stock II
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times). However, you may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).

```javascript
var maxProfit = function(prices) {
    let profit = 0;

    for (let i = 0; i < prices.length; i++) {
        if (prices[i+1]>prices[i]) profit += prices[i+1]-prices[i];
    }

    return profit;
};
```

If the next price is larger than the current price, add their difference to the profit.

## 53 Maximum Subarray

Find the contiguous subarray within an array (containing at least one number) which has the largest sum.

For example, given the array `[-2,1,-3,4,-1,2,1,-5,4]`,
the contiguous subarray `[4,-1,2,1]` has the largest sum = `6`.

click to show more practice.

More practice:
If you have figured out the `O(n)` solution, try coding another solution using the divide and conquer approach, which is more subtle.


```javascript
/**
 * @param {number[]} nums
 * @return {number}
 */
var maxSubArray = function(nums) {

    let maxSum = nums[0];
    let localSum = 0;

    for (let i = 0; i < nums.length; i++) {


       //accumulate
        localSum += nums[i];

        //update maxSum if current num improves it
        maxSum = Math.max(localSum, maxSum);

        //reset local if it goes below 0
        localSum = Math.max(localSum, 0);
    }

    return maxSum;
};
```

## *69 Sqrt(x)

Implement `int sqrt(int x)`.

Compute and return the square root of `x`.

```javascript
/**
 * @param {number} x
 * @return {number}
 */
var mySqrt = function(x) {
    if (x < 1) return 0;

    let left = 1,
        right = x,
        mid;

    while (left <= right) {
        mid = Math.floor((right - left)/2 + left);

        if (mid > x / mid) { /* same as mid * mid > x but prevents overflow */
            right = mid - 1 /* answer <= (mid - 1) */
        } else {
            left = mid + 1
        }
    }

    return right;

};
```

Use a binary search to find the sqaure root. We find the last number for which (m * m <= x) is true.

## 42 Trapping Rain Water

Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.

For example,
Given ```[0,1,0,2,1,0,1,3,2,1,2,1]```, return ```6```.

![enter image description here](http://www.leetcode.com/static/images/problemset/rainwatertrap.png)
The above elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.


```javascript
/**
 * @param {number[]} height
 * @return {number}
 */
var trap = function(height) {

    let left = 0,
        right = height.length - 1,
        leftMax = 0,
        rightMax = 0,
        rainWater = 0;

    while (left < right) {
        if (height[left] < height[right]) {
            height[left] >= leftMax ? (leftMax = height[left]) : rainWater += (leftMax - height[left]);
            ++left;
        } else {
            height[right] >= rightMax ? (rightMax = height[right]) : rainWater += (rightMax - height[right]);
            --right;
        }
    }

    return rainWater;

};
```

Second Method:
```javascript
var trap = function(height) {

   let globalMax = height[0];
    let globalMaxIndex = 0;
    let rainWater = 0;

    for (let i = 1; i < height.length; i++){
        if (height[i] > globalMax) {
            globalMax = height[i];
            globalMaxIndex = i;
        }
    }

    let leftMax = 0;

    for (let i = 0; i < globalMaxIndex; i++){
        if (height[i] >= leftMax){
            leftMax = height[i];
        } else {
            rainWater += leftMax - height[i];
        }
    }

    let rightMax = 0;

    for (let i = height.length - 1; i > globalMaxIndex; i--){
        if (height[i] >= rightMax){
            rightMax = height[i];
        } else {
            rainWater += rightMax - height[i];
        }
    }

    return rainWater;

};
```
For the second method, we find the global max of the height array. Then we iterate through the array from left to the global max and from right to the global max. In each case the amount of rain water that is trapped is dictated by the preceding local maxes. On the left side of the global max the left wall will dictate the height of the rain water while on the right side the right wall will dictate. To calculate the water subtract the current height from the height of the previous max (the bounding wall other than the global max wall).

## 232 Implement Queue using Stacks
Implement the following operations of a queue using stacks.

`push(x)` -- Push element x to the back of queue.
`pop()` -- Removes the element from in front of queue.
`peek()`-- Get the front element.
`empty()` -- Return whether the queue is empty.
Notes:

 - You must use only standard operations of a stack -- which means only `push to top`, `peek/pop from top`, `size`, and `is empty` operations are valid.
 - Depending on your language, stack may not be supported natively. You may simulate a stack by using a list or deque (double-ended queue), as long as you use only standard operations of a stack.
 - You may assume that all operations are valid (for example,
   no pop or peek operations will be called on an empty queue).

```javascript
/**
 * Initialize your data structure here.
 */
var MyQueue = function() {
    this.stackA = [];
    this.stackB = [];
};

/**
 * Push element x to the back of queue.
 * @param {number} x
 * @return {void}
 */
MyQueue.prototype.push = function(x) {
    //always push onto stackA
    this.stackA.push(x)
};

/**
 * Removes the element from in front of queue and returns that element.
 * @return {number}
 */
MyQueue.prototype.pop = function() {
    if (this.stackB.length){
        return this.stackB.pop();
    } else {
        while (this.stackA.length) {
            this.stackB.push(this.stackA.pop());
        }
        return this.stackB.pop();
    }
};

/**
 * Get the front element.
 * @return {number}
 */
MyQueue.prototype.peek = function() {
    if (this.stackB.length){
        return this.stackB[this.stackB.length - 1];
    } else {
        while (this.stackA.length) {
            this.stackB.push(this.stackA.pop());
        }
        return this.stackB[this.stackB.length - 1];
    }
};

/**
 * Returns whether the queue is empty.
 * @return {boolean}
 */
MyQueue.prototype.empty = function() {
    return !this.stackA.length && !this.stackB.length;
};
```

We use two stacks - an instack and an outstack. We always push on the instack and pop from the outstack. If the outstack has length we simply pop from it, but if it doesnt then we pop from the instack, push onto the outstack and then pop from the outstack. This process reverses the order of insertion and simulates a queue (FIFO).

## 386 Lexicographical Numbers
Given an integer `n`, return `1 - n` in lexicographical order.

For example, given 13, return: `[1,10,11,12,13,2,3,4,5,6,7,8,9]`.

Please optimize your algorithm to use less time and space. The input size may be as large as `5,000,000`.

```javascript
/**
 * @param {number} n
 * @return {number[]}
 */
var lexicalOrder = function(n) {
    let lexArr = [];

    for (let i = 1; i < 10; i++) {
        depthFirstSearch(lexArr, n, i);
    }

    return lexArr;

};

function depthFirstSearch(lexArr, n, currentNum) {
    if (currentNum > n) {
        return;
    }
    else {
        //add currentNum to lexArr
        lexArr.push(currentNum);

        for (let i = 0; i < 10; i++){
            //this line is not necessary but avoids some duplicate calculations
            if (10 * currentNum + i > n) return;

            depthFirstSearch(lexArr, n, 10 * currentNum + i)
        }


    }
}
```

We create our lexicographically ordered array in a depth first way - going from 1 to 10 to 100 and continually shift left until the currentNum is larger than n. For n = 45 we go from 1 to 10 to 100 which is greater than 45 so we return and continue with 11, 12, 13... and so on. We then buble back up to change the leftmost digit.

## 50 Pow(x, n)

Implement pow(x, n).
```javascript
/**
 * @param {number} x
 * @param {number} n
 * @return {number}
 */
var myPow = function(x, n) {

   if (n == 0) return 1;
   if (n < 0){
            n = -n;
            x = 1/x;
    }
    return (n % 2 == 0) ? myPow(x*x, n/2) : x*myPow(x*x, (n - 1)/2);
};
```

If n is even we call myPow with x*x and divide n by 2. If n is odd we do the same except we also subtract 1 and muliply the result by x to account for the subtraction. This takes advantage of the property that x ^ n = (x * x) ^ n / 2.
For x = 2 and n = 5 we get:
2 * myPow(4, 2) --> 2 * myPow(16, 1) --> 2 * 16 * myPow(256, 0) --> 2 * 16 * 1 --> 32

If n is less than 0 then we flip its sign and reassign x to 1/x. 2^ -2 = 1/2 ^ 2 = 1/4 = 0.25

## 56 Merge Intervals

Given a collection of intervals, merge all overlapping intervals.

For example,
Given `[1,3],[2,6],[8,10],[15,18]`,
return `[1,6],[8,10],[15,18]`

## 287 Find the Duplicate Number

Given an array nums containing `n + 1` integers where each integer is between `1` and `n` (inclusive), prove that at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate one.

## 2 Add Two Numbers
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

## 268 Missing Number
Given an array containing `n` distinct numbers taken from `0, 1, 2, ..., n`, find the one that is missing from the array.

For example,
Given nums = `[0, 1, 3]` return 2.

Note:
Your algorithm should run in linear runtime complexity. Could you implement it using only constant extra space complexity?

## 98 Validate Binary Search Tree

Given a binary tree, determine if it is a valid binary search tree (BST).

Assume a BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.
**Example 1:**
```
    2
   / \
  1   3
```
Binary tree `[2,1,3]`, return **true**.
**Example 2:**
```
    1
   / \
  2   3
```
Binary tree `[1,2,3]`, return **false**.

```javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @return {boolean}
 */
var isValidBST = function(root) {

    let prev = null;
    let currentNode = root;
    let stack = [];

    while (stack.length || currentNode != null) {
        if (currentNode) {
            stack.push(currentNode);
            currentNode = currentNode.left;
        } else {
            let p = stack.pop() ;
		    if (prev != null && p.val <= prev.val) {
			    return false ;
		    }
		    prev = p;
			currentNode = p.right;

        }
    }

    return true;


};
```

```javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @return {boolean}
 */
var isValidBST = function(root, upperBound = Infinity, lowerBound = -Infinity) {

    if (root === null) return true;
    if (root.val <= lowerBound || root.val >= upperBound) return false;

    return isValidBST(root.left, root.val, lowerBound) && isValidBST(root.right, upperBound, root.val);

};
```

## 151 Reverse Words in a String

Given an input string, reverse the string word by word.

For example,
Given s = `"the sky is blue"`,
return `"blue is sky the"`.

Update (2015-02-12):
For C programmers: Try to solve it in-place in O(1) space.

## 215 Kth Largest Element in an Array
Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

For example,
Given `[3,2,1,5,6,4]` and `k = 2`, return `5`.

Note:
You may assume k is always valid, 1 ≤ k ≤ array's length.

## 215 Kth Largest Element in an Array
Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

For example,
Given `[3,2,1,5,6,4]` and `k = 2`, return `5`.

Note:
You may assume k is always valid, 1 ≤ k ≤ array's length.

```javascript
/**
 * @param {number[]} nums
 * @param {number} k
 * @return {number}
 */
var findKthLargest = function(nums, k) {
    nums = nums.sort ((a, b) => a - b);
    return nums[nums.length - k]
};
```

## 88 Merge Sorted Array
Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

Note:
You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2. The number of elements initialized in nums1 and nums2 are m and n respectively.


```javascript
/**
 * @param {number[]} nums1
 * @param {number} m
 * @param {number[]} nums2
 * @param {number} n
 * @return {void} Do not return anything, modify nums1 in-place instead.
 */
var merge = function(nums1, m, nums2, n) {

    let nums1_index = m - 1;
    let nums2_index = n - 1;
    let new_tail = m + n - 1;

    while (nums2_index >= 0) {
        if (nums1_index >= 0 && nums1[nums1_index] > nums2[nums2_index]) {
            nums1[new_tail] = nums1[nums1_index];
            nums1_index--;
        } else {
            nums1[new_tail] = nums2[nums2_index];
            nums2_index--;
        }
        new_tail--;
    }
};
```

nums1 will always have enough space to store both sets of numbers. To not overwrite any values in nums1 we start from the back of the array. We continue to store numbers in nums1 until the numes2_index is less than 0 and has run off the array. O(n + m)

## 49 Group Anagrams
Given an array of strings, group anagrams together.

For example, given: `["eat", "tea", "tan", "ate", "nat", "bat"]`,
Return:
[
  ["ate", "eat","tea"],
  ["nat","tan"],
  ["bat"]
]

```javascript
/**
 * @param {string[]} strs
 * @return {string[][]}
 */
var groupAnagrams = function(strs) {

    let anagrams = {}

    strs.forEach((word) => {
        let signature = word.split('').sort((a, b) => a.localeCompare(b)).join('');
        anagrams[signature] ? anagrams[signature].push(word) : anagrams[signature] = [word];
    })

    return Object.values(anagrams)
};
```

## 102. Binary Tree Level Order Traversal

Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

For example:
Given binary tree `[3,9,20,null,null,15,7]`,
```
    3
   / \
  9  20
    /  \
   15   7
```
return its level order traversal as:
```
[
  [3],
  [9,20],
  [15,7]
]
```

```javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number[][]}
 */
var levelOrder = function(root) {

    if (!root) return [];

    let queue = [root];
    let levelArr = [];


    while(queue.length) {

        let level = [];
        let size = queue.length;

        for (let i = 0; i < size; i++){
            let currentNode = queue.shift();
            level.push(currentNode.val)
            if (currentNode.left) queue.push(currentNode.left);
            if (currentNode.right) queue.push(currentNode.right);
        }

        levelArr.push(level);
    }

    return levelArr;

};
```

## 225. Implement Stack using Queues

Implement the following operations of a stack using queues.

push(x) -- Push element x onto stack.
pop() -- Removes the element on top of the stack.
top() -- Get the top element.
empty() -- Return whether the stack is empty.
**Notes:**

 1. You must use only standard operations of a queue -- which means only `push to back`, `peek/pop from front`, `size`, and `is empty` operations are valid.
 2. Depending on your language, queue may not be supported natively. You may simulate a queue by using a list or  deque (double-ended queue), as long as you use only standard operations of a queue.
 3. You may assume that all operations are valid (for example, no pop or top operations will be called on an empty stack).


```javascript
/**
 * Initialize your data structure here.
 */
var MyStack = function() {
    this.queue = [];
};

/**
 * Push element x onto stack.
 * @param {number} x
 * @return {void}
 */
MyStack.prototype.push = function(x) {
    this.queue.push(x);
    for (let i = 1; i < this.queue.length; i++) {
            this.queue.push(this.queue.shift());

        }
};

/**
 * Removes the element on top of the stack and returns that element.
 * @return {number}
 */
MyStack.prototype.pop = function() {
    return this.queue.shift();
};

/**
 * Get the top element.
 * @return {number}
 */
MyStack.prototype.top = function() {
   return this.queue[0];
};

/**
 * Returns whether the stack is empty.
 * @return {boolean}
 */
MyStack.prototype.empty = function() {
    return !this.queue.length;
};
```
For every push onto the stack we reverse the queue with `this.queue.push(this.queue.shift());` This is O(n) for push and constant for every other operation.

## 3. Longest Substring Without Repeating Characters

Given a string, find the length of the longest substring without repeating characters.

Examples:

Given `"abcabcbb"`, the answer is `"abc"`, which the length is 3.

Given `"bbbbb"`, the answer is `"b"`, with the length of 1.

Given `"pwwkew"`, the answer is `"wke"`, with the length of 3. Note that the answer must be a substring, `"pwke"` is a subsequence and not a substring.

```javascript
/**
 * @param {string} s
 * @return {number}
 */
var lengthOfLongestSubstring = function(s) {

    let seen = new Set();
    let subStart = 0;
    let subEnd = 0;
    let length = 0;

    while (subEnd < s.length) {
        if (!seen.has(s[subEnd])) {
            seen.add(s[subEnd]);
            subEnd++;
            length = Math.max(length, seen.size);

        } else {
            seen.delete(s[subStart]);
            subStart++;
        }
    }

    return length;
};
```
We use a set to track the longest substring without repeating characters so far. If the s[subEnd] is not in the set, add it to the set and increment subEnd. Update the length if seen.size is larger than the length. Else s[subEnd] is in the set and needs to be removed. While its still in the set we remove s[subStart] and incrment subStart until it is not in the set.

## 103 Binary Tree Zigzag Level Order Traversal

Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).

For example:
Given binary tree `[3,9,20,null,null,15,7]`,
```
    3
   / \
  9  20
    /  \
   15   7
```
return its zigzag level order traversal as:
```
[
  [3],
  [20,9],
  [15,7]
]
```

```javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number[][]}
 */
var zigzagLevelOrder = function(root) {

    if (!root) return [];

    let queue = [root];
    let zigZag = [];
    let isReverse = false;

    while (queue.length) {

        let level = [];
        let length = queue.length;

        for (let i = 0; i < length; i++) {

            currentNode = queue.shift();
            isReverse ? level.unshift(currentNode.val) : level.push(currentNode.val);

            if (currentNode.left) queue.push(currentNode.left);
            if (currentNode.right) queue.push(currentNode.right);
        }

        zigZag.push(level);
        isReverse = !isReverse;
    }

    return zigZag;
};
```

Implement a level order traversal. Based on the level we either push or shift onto the level array. 

