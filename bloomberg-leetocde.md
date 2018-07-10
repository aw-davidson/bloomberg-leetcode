## 62 Unique Paths
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?
![enter image description here](https://leetcode.com/static/images/problemset/robot_maze.png)
Above is a 3 x 7 grid. How many possible unique paths are there?

Note: m and n will be at most 100.

```javascript
/**
 * @param {number} m
 * @param {number} n
 * @return {number}
 */
var uniquePaths = function(m, n) {
    let currentRow = new Array(n);
    // Assigning a 1 to matrix[0][0] is simply a shortcut that skips some later computation
    // as matrix[i][0] will never change in this iterative process
    for (let i = 0; i < n; i++) {
        currentRow[i] = 1;
    }
    for (let row = 1; row < m; row++) {
        for (let i = 1; i < n; i++) {
            currentRow[i] += currentRow[i - 1];
        }
    }
    return currentRow[n - 1];
};
```
## 63. Unique Paths II
Follow up for "Unique Paths" (62):

Now consider if some obstacles are added to the grids. How many unique paths would there be?

An obstacle and empty space is marked as `1` and `0` respectively in the grid.

For example,
There is one obstacle in the middle of a 3x3 grid as illustrated below.
```
[
  [0,0,0],
  [0,1,0],
  [0,0,0]
]
```
The total number of unique paths is `2`.

**Note**: m and n will be at most `100`.
```javascript
/**
 * @param {number[][]} obstacleGrid
 * @return {number}
 */
var uniquePathsWithObstacles = function(obstacleGrid) {
    const cols = obstacleGrid[0].length;

    const paths = Array.apply(null, Array(cols)).map(() => 0);
    paths[0] = 1;

    obstacleGrid.forEach((currentRow) => {
        currentRow.forEach((element, colIndex) => {
            if (element == 1) {
                paths[colIndex] = 0;
            } else if (colIndex > 0) {
                paths[colIndex] = paths[colIndex] + paths[colIndex - 1];
            }
        });
    });

    return paths[cols - 1];
};
```

## 146 LRU Cache
Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and put.

```get(key)``` - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return ```-1```.
```put(key, value)``` - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.

Follow up:
Could you do both operations in ```O(1)``` time complexity?


```javascript
var LRUCache = function(capacity) {
    this.capacity = capacity;
    this.map = new Map();
};

/**
 * @param {number} key
 * @return {number}
 */
LRUCache.prototype.get = function(key) {
  if (this.map.has(key)) {
      let val = this.map.get(key);
      this.map.delete(key);
      this.map.set(key, val);
      return this.map.get(key);
  }

  return -1;

};

/**
 * @param {number} key
 * @param {number} value
 * @return {void}
 */
LRUCache.prototype.put = function(key, value) {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, value);
    let keys = this.map.keys();
    while (this.map.size > this.capacity) this.map.delete(keys.next().value)

};
```

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

```javascript
function merge(meetings) {

  // Create a deep copy of the meetings array
  // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/assign#Deep_Clone
  const meetingsCopy = JSON.parse(JSON.stringify(meetings));


  // Sort by start time
  const sortedMeetings = meetings.sort((a, b) => {
    return a.startTime - b.startTime;
  });

  // Initialize mergedMeetings with the earliest meeting
  const mergedMeetings = [sortedMeetings[0]];

  for (let i = 1; i < sortedMeetings.length; i++) {
    const currentMeeting    = sortedMeetings[i];
    const lastMergedMeeting = mergedMeetings[mergedMeetings.length - 1];

    // If the current meeting overlaps with the last merged meeting, use the
    // later end time of the two
    if (currentMeeting.startTime <= lastMergedMeeting.endTime) {
      lastMergedMeeting.endTime = Math.max(lastMergedMeeting.endTime, currentMeeting.endTime);
    } else {

      // Add the current meeting since it doesn't overlap
      mergedMeetings.push(currentMeeting);
    }
  }

  return mergedMeetings;
}
```

## 16. 3Sum Closest

Given an array S of n integers, find three integers in S such that the sum is closest to a given number, target. Return the sum of the three integers. You may assume that each input would have exactly one solution.

    For example, given array S = {-1 2 1 -4}, and target = 1.

    The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).

```javascript
var threeSumClosest = function(nums, target) {

    let closestSum = nums[0] + nums[1] + nums[2];
    nums = nums.sort((a, b) => a - b)

    for (let i = 0; i < nums.length - 2; i++) {
        let lo = i + 1;
        let hi = nums.length - 1;

        while (lo < hi) {
            let sum = nums[i] + nums[lo] + nums[hi];

            if (Math.abs(closestSum - target) > Math.abs(sum - target)) {
                closestSum = sum;
            }

            if (sum === target) return closestSum;
            if (sum < target) lo++;
            if (sum > target) hi--;

        }

    }

    return closestSum;
};
```

## 287 Find the Duplicate Number

Given an array nums containing `n + 1` integers where each integer is between `1` and `n` (inclusive), prove that at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate one.

```javascript
var findDuplicate = function(nums) {
    let slow = nums[0], fast = nums[nums[0]];

    //here fast moves two steps at a time
    while(slow!==fast){
    	slow = nums[slow];
    	fast = nums[nums[fast]];
    }

    fast=0;

    //here fast move one step at a time
    while(slow!==fast){
    	slow = nums[slow];
    	fast = nums[fast];
    }
    return slow;
};
```

The array of values of n + 1 can be thought of as a linked list where the value at each index is a pointer to another index. This problem is the same as a linked list with a cycle.

## 2 Add Two Numbers
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

```javascript
var addTwoNumbers = function(l1, l2) {
        let s1 = [];
        let s2 = [];

        while(l1 != null) {
            s1.push(l1.val);
            l1 = l1.next;
        };
        while(l2 != null) {
            s2.push(l2.val);
            l2 = l2.next;
        }

        let sum = 0;
        let currentNode = new ListNode(0);

        while (s1.length || s2.length) {
            if (s1.length) sum += s1.pop();
            if (s2.length) sum += s2.pop();


            //if sum is 15 then val is 5
            currentNode.val = sum % 10;


            let nextNode = new ListNode(Math.floor(sum / 10));
            nextNode.next = currentNode;

            currentNode = nextNode;
            sum = Math.floor(sum / 10);

        }

        //if the first significant digits sum to larger than 9 and need a carry
        return currentNode.val ? currentNode : currentNode.next ;
};
```

Iterate through both linked lists to create a stack for each. Pop off of the stack and add the two numbers to the sum. Create new nodes with the first significant digit of the sum by dividing by 10 - this insures that the first digit of the sum is correct. shift the sum right by dividing by 10.

## 268 Missing Number
Given an array containing `n` distinct numbers taken from `0, 1, 2, ..., n`, find the one that is missing from the array.

For example,
Given nums = `[0, 1, 3]` return 2.

Note:
Your algorithm should run in linear runtime complexity. Could you implement it using only constant extra space complexity?

```javascript
var missingNumber = function(nums) {
    let n = nums.length;
    let sumWithMissing = ((n * n) + n)/2;
    let actualSum = nums.reduce((acc, curr) => acc + curr, 0);

    return sumWithMissing - actualSum;
};
```

use a triangular series.

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

```javascript
const findKthLargest = (nums, k) => {
    return quickSelect(nums, 0, nums.length - 1, k);
};

const quickSelect = (nums, lo, hi, k) => {
    // use quick sort's idea
    // put nums that are <= pivot to the left
    // put nums that are  > pivot to the right
    for (var i = lo, j = lo; j < hi; j++) {
        if (nums[j] <= nums[hi]) {
            swap(nums, i++, j);
        }
    }
    swap(nums, i, j);

    // count the nums that are >= pivot
    const m = hi - i + 1;
    // pivot is the one!
    if (m === k) return nums[i];
    // pivot is too small, so it must be on the right
    if (m > k) return quickSelect(nums, i + 1, hi, k);
    // pivot is too big, so it must be on the left
    return quickSelect(nums, lo, i - 1, k - m);
};

function swap(nums, i, j) {
    [nums[i], nums[j]] = [nums[j], nums[i]]
}
```



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

## 26. Remove Duplicates from Sorted Array

Given a sorted array, remove the duplicates in place such that each element appear only once and return the new length.

Do not allocate extra space for another array, you must do this in place with constant memory.

For example,
Given input array nums = `[1,1,2]`,

Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively. It doesn't matter what you leave beyond the new length.

```javascript
/**
 * @param {number[]} nums
 * @return {number}
 */
var removeDuplicates = function(nums) {

    //first elem will always be unique
    let trueIndex = 1;

    for (let i = 1; i < nums.length; i++) {
        if (nums[i] !== nums[i - 1]) {
            nums[trueIndex] = nums[i];
            trueIndex++;
        }
    }

    //length
    return trueIndex;
};
```

A number is unique if the element adjacent to it is different. Iterate through the array checking for numbers that are unique then add the unique numbers at the trueIndex. The trueIndex will overwite values that are duplicates.

## 105. Construct Binary Tree from Preorder and Inorder Traversal

Given preorder and inorder traversal of a tree, construct the binary tree.
**Note:**
You may assume that duplicates do not exist in the tree.

```javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {number[]} preorder
 * @param {number[]} inorder
 * @return {TreeNode}
 */
var buildTree = function(preorder, inorder) {
    if (preorder.length == 0) return null;

    var root = new TreeNode(preorder[0]);
    var i = 0;
    while (i < inorder.length && inorder[i] != preorder[0]) i++;

    // divide inorder list into left and right;
    root.left = buildTree(preorder.slice(1, i+1), inorder.slice(0, i));
    root.right = buildTree(preorder.slice(i+1, preorder.length), inorder.slice(i+1, inorder.length));

    return root;

};

```

            1
           / \
          2   3
         / \   \
        4   5   6
the preorder and inorder arrays are:

preorder: 1 2 4 5 3 6
inorder: 4 2 5 1 3 6

let's group the left nodes with ( ) and right nodes with [ ]:

preorder: 1 (2 4 5) [3 6]  //left sub tree goes from index 1 to i + 1 - right is i + 1 to end
inorder: (4 2 5) 1 [3 6]

we can then build the left subtree using the following preorder and inorder arrays:

preorder: 2 4 5
inorder: 4 2 5

and for the right subtree:

preorder: 3 6
inorder: 3 6

## 79. Word Search

Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

For example,
Given board =
```
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
```

word = `"ABCCED"`, -> returns `true`,
word = `"SEE"`, -> returns `true`,
word = `"ABCB"`, -> returns `false`.


```javascript
/**
 * @param {character[][]} board
 * @param {string} word
 * @return {boolean}
 */
function exist(board, word) {

  function find(i, j, k) { //k is the index which keeps track of where we are with the word
    //if i or j are not within the bounds of the board
    if (i < 0 || j < 0 || i > board.length - 1 || j > board[0].length - 1) return false;

    //if the current letter is not equal to k
    if (board[i][j] !== word[k]) return false;

    //if we have returned true for all previous letters and are now at the end of the word
    if (k === word.length - 1) return true;

    board[i][j] = '*';      // mark as visited

    if (find(i - 1, j, k + 1)) return true;  // up
    if (find(i + 1, j, k + 1)) return true;  // down
    if (find(i, j - 1, k + 1)) return true;  // left
    if (find(i, j + 1, k + 1)) return true;  // right

    board[i][j] = word[k];  // reset
    return false;
  }

  for (let i = 0; i < board.length; i++) {
    for (let j = 0; j < board[i].length; j++) {
      if (find(i, j, 0)) return true;
    }
  }

  return false;
}
```



## 141. Linked List Cycle

Given a linked list, determine if it has a cycle in it.

Follow up:
Can you solve it without using extra space?

```javascript
/**
 * Definition for singly-linked list.
 * function ListNode(val) {
 *     this.val = val;
 *     this.next = null;
 * }
 */

/**
 * @param {ListNode} head
 * @return {boolean}
 */
var hasCycle = function(head) {
       // start both runners at the beginning
    var slowRunner = head;
    var fastRunner = head;

    // until we hit the end of the list
    while (fastRunner && fastRunner.next) {
        slowRunner = slowRunner.next;
        fastRunner = fastRunner.next.next;

        // case: fastRunner is about to "lap" slowRunner
        if (fastRunner === slowRunner) {
            return true;
        }
    }

    // case: fastRunner hit the end of the list
    return false;
};
```

## 5. Longest Palindromic Substring

Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

**Example:**
```
Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer
```
**Example:**

```
Input: "cbbd"

Output: "bb"
```


```javascript
/**
 * @param {string} s
 * @return {string}
 */
var longestPalindrome = function(s) {

    let longest = "";

    for (let i = 0; i < s.length; i++) {
        let odd = expand(s, i, i);
        let even = expand(s, i, i + 1);

        if (odd.length > longest.length) longest = odd;
        if (even.length > longest.length) longest = even;


    }

    return longest;
}

function expand(s, j, k) {

   while (j >= 0 && k < s.length && s[j] == s[k]) {
		j--;
		k++;
	}

    return s.slice(j + 1, k);
}

```
## 13. Roman to Integer

Given a roman numeral, convert it to an integer.

Input is guaranteed to be within the range from 1 to 3999.

**Notes:**
canonical numbers (numbers described by a single letter):
I = 1
V = 5
X = 10
L = 50
C = 100
D = 500
M = 1000

Additive rule: Use left to right descending value canonical numbers to represent number
e.g. XVII = 17

Subtractive rule: In case additive rule returns more than 4 same characters in a row, write next larger canonical numeral and prefix numeral sequence to subtract.
e.g. IIII = 4 is written as IV (5-1)

```javascript
/**
 * @param {string} s
 * @return {number}
 */
var romanToInt = function(s) {
    let integer = 0;
    let hash = {
        I:1,
        V:5,
        X:10,
        L:50,
        C:100,
        D:500,
        M:1000
    }

    for (let i = 0; i < s.length; i++){

        let current = hash[s[i]];
        let next = s[i + 1] ? hash[s[i + 1]] : s[i];

        if (current < next) {
            integer += next - current;
            i++;
        } else {
            integer += current;
        }
    }

    return integer;
};
```

The only case when you subtract is when the next numeral is larger than the previous.

## 100. Same Tree
Given two binary trees, write a function to check if they are equal or not.

Two binary trees are considered equal if they are structurally identical and the nodes have the same value.

```javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} p
 * @param {TreeNode} q
 * @return {boolean}
 */
var isSameTree = function(p, q) {

	if(p===null && q==null) return true;
	if(p===null || q===null) return false;
	return p.val===q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
};
```
## 139. Word Break

Given a **non-empty** string s and a dictionary wordDict containing a list of **non-empty** words, determine if s can be segmented into a space-separated sequence of one or more dictionary words. You may assume the dictionary does not contain duplicate words.

For example, given
s = `"leetcode"`,
dict = `["leet", "code"]`.

Return true because `"leetcode"` can be segmented as `"leet code"`.

```javascript
var wordBreak = function(s, wordDict) {
    if (wordDict.length === 0) return false;
    if (wordDict.length === 1) return s === wordDict[0];

    let queue = [''];
    let memo = new Map();

    while (queue.length > 0) {
        const val = queue.shift();

        for (let word of wordDict) {
            const searchWord = `${val}${word}`;
            const startsWith = s.indexOf(searchWord) === 0;


            if (searchWord === s) return true;

            else if (!memo.has(searchWord) && startsWith) {
                memo.set(searchWord, true);
                queue.push(searchWord);
            }
        }
    }

    return false;
};
```

We do a breadfirst search with memoization. Nodes represent a substring of s that can be made up of words from the wordDict. A path from the root to a leaf node that is s represents a way to compose that word with words from the wordDict.

## 15. 3Sum

Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

Note: The solution set must not contain duplicate triplets.
```js
For example, given array S = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```

```javascript
/**
 * @param {number[]} nums
 * @return {number[][]}
 */
var threeSum = function(nums) {

  let triplets = [];
  nums = nums.sort((a, b) => a - b);

  for(let i = 0; i < nums.length-2; i++) {
    if(i == 0 || nums[i] > nums[i - 1]) {
      let lo = i + 1;
      let high = nums.length - 1;

      while(lo < high) {
        let sum = nums[i] + nums[lo] + nums[high];
        if(sum == 0) {
          triplets.push([nums[i], nums[lo], nums[high]]);
          lo++;
          high--;
          //skip duplicates from lo
          while(lo<high && nums[lo]==nums[lo-1])
            lo++;

          //skip duplicates from high
          while(lo<high && nums[high]==nums[high+1])
            high--;
        } else if(sum < 0) {
          lo++;
        } else {
          high--;
        }
      }
    }
  }

  return triplets;
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
var isSymmetric = function(root) {
   return root==null || isSymmetricHelp(root.left, root.right);
};

function isSymmetricHelp(left, right){

    if(left==null || right==null) return left === right;

    if(left.val !== right.val) return false;

    return isSymmetricHelp(left.left, right.right) && isSymmetricHelp(left.right, right.left);
}
```

Alternatively, you can perform a level-order traversal and check to see if each level is symmetrical.

# 172. Factorial Trailing Zeroes
Given an integer n, return the number of trailing zeroes in n!.

**Note:** Your solution should be in logarithmic time complexity.


```javascript
/**
 * @param {number} n
 * @return {number}
 */
var trailingZeroes = function(n) {
    return n == 0 ? 0 : Math.floor(n / 5) + trailingZeroes(Math.floor(n / 5));
};
```

alternate solution:

```javascript
/**
 * @param {number} n
 * @return {number}
 */
var trailingZeroes = function(n) {
    let result = 0;
    for (let i=5; Math.floor(n/i) > 0; i*=5){
        result += Math.floor(n/i);
    }
    return result;
};
```
A traling zero occurs when 10 occurs in the n! sequence. 10 comes from 5 and 2 in n! If we match all of the 5s with all of the 2s then we can count the matches to get the trailing 0 count. Additionally, since even numbers are abundant in n! we just need to count the 5s and we can assume that we will always have a matching 2 from one even number.

Example One

How many multiples of 5 are between 1 and 23? There is 5, 10, 15, and 20, for four multiples of 5. Paired with 2's from the even factors, this makes for four factors of 10, so: 23! has 4 zeros.

Example Two

How many multiples of 5 are there in the numbers from 1 to 100?

because 100 ÷ 5 = 20, so, there are twenty multiples of 5 between 1 and 100.

but wait, actually 25 is 5×5, so each multiple of 25 has an extra factor of 5, e.g. 25 × 4 = 100，which introduces extra of zero.

So, we need know how many multiples of 25 are between 1 and 100? Since 100 ÷ 25 = 4, there are four multiples of 25 between 1 and 100.

Finally, we get 20 + 4 = 24 trailing zeroes in 100!

The above example tell us, we need care about 5, 5×5, 5×5×5, 5×5×5×5 ....

Example Three

By given number 4617.

5^1 : 4617 ÷ 5 = 923.4, so we get 923 factors of 5

5^2 : 4617 ÷ 25 = 184.68, so we get 184 additional factors of 5

5^3 : 4617 ÷ 125 = 36.936, so we get 36 additional factors of 5

5^4 : 4617 ÷ 625 = 7.3872, so we get 7 additional factors of 5

5^5 : 4617 ÷ 3125 = 1.47744, so we get 1 more factor of 5

5^6 : 4617 ÷ 15625 = 0.295488, which is less than 1, so stop here.

Then 4617! has 923 + 184 + 36 + 7 + 1 = 1151 trailing zeroes.


## 24. Swap Nodes in Pairs

Given a linked list, swap every two adjacent nodes and return its head.

For example,
Given `1->2->3->4`, you should return the list as `2->1->4->3`.

Your algorithm should use only constant space. You may not modify the values in the list, only nodes itself can be changed.

```javascript
/**
 * Definition for singly-linked list.
 * function ListNode(val) {
 *     this.val = val;
 *     this.next = null;
 * }
 */
/**
 * @param {ListNode} head
 * @return {ListNode}
 */
var swapPairs = function(head) {

        if (head == null || head.next == null) {
            return head;
        }

        let first = head,
            last = head.next;

        first.next = swapPairs(last.next);
        last.next = first;
        return last;
};
```
swapPairs is called with the 'first' node and returns the 'last' node. If there is no 'last' node, then it will return the first.

## 230. Kth Smallest Element in a BST

Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

Note:
You may assume k is always valid, 1 ? k ? BST's total elements.

Follow up:
What if the BST is modified (insert/delete operations) often and you need to find the kth smallest frequently? How would you optimize the kthSmallest routine?


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
 * @param {number} k
 * @return {number}
 */
var kthSmallest = function(root, k) {
    let traversal = [];
    return traversal[k - 1];
};

function inorder(root, traversal) {
    if (root.left) inorder(root.left, traversal);
    traversal.push(root.val);
    if (root.right) inorder(root.right, traversal);
}
```

iterative solution:

```javascript
var kthSmallest = function(root, k) {
    let stack = [];
    let count = 0;
    let node = root;

    while (true){
        if (node){
            stack.push(node);
            node = node.left;
        } else {
            if (stack.length == 0) break;
            node = stack.pop();
            count += 1;
            if (count == k) return node.val;
            node = node.right;
        }
    }
};
```

## 274. H-Index

Given an array of citations (each citation is a non-negative integer) of a researcher, write a function to compute the researcher's h-index.

According to the definition of h-index on Wikipedia: "A scientist has index h if h of his/her N papers have at least h citations each, and the other N − h papers have no more than h citations each."

For example, given `citations = [3, 0, 6, 1, 5]`, which means the researcher has `5` papers in total and each of them had received `3, 0, 6, 1, 5` citations respectively. Since the researcher has `3` papers with at least `3` citations each and the remaining two with no more than `3` citations each, his h-index is `3`.

Note: If there are several possible values for `h`, the maximum one is taken as the `h`-index.

```javascript

/**
 * @param {number[]} citations
 * @return {number}
 */
var hIndex = function(citations) {

    let n = citations.length;
    let buckets = Array.apply(null, Array(n+1)).map(() => 0);

    for (let i = 0; i < n; i++){
        if (citations[i] > n) {
            buckets[n]++;
        } else {
            buckets[citations[i]]++;
        }
    }

    let count = 0;
    for(let i = n; i >= 0; i--) {
        count += buckets[i];
        if(count >= i) {
            return i;
        }
    }
    return 0;

};
```


So assume n is the total number of papers, if we have n+1 buckets, number from 0 to n, then for any paper with reference corresponding to the index of the bucket, we increment the count for that bucket. The only exception is that for any paper with larger number of reference than n, we put in the n-th bucket.

Then we iterate from the back to the front of the buckets, whenever the total count exceeds the index of the bucket, meaning that we have the index number of papers that have reference greater than or equal to the index. Which will be our h-index result. The reason to scan from the end of the array is that we are looking for the greatest h-index.

## 11. Container With Most Water

Given n non-negative integers $a_1, a_2,..., a_n$, where each represents a point at coordinate $(i, a_i)$. n vertical lines are drawn such that the two endpoints of line $i$ is at $(i, a_i)$ and $(i, 0)$. Find two lines, which together with x-axis forms a container, such that the container contains the most water.

Note: You may not slant the container and n is at least 2.


```javascript

/**
 * @param {number[]} height
 * @return {number}
 */
var maxArea = function(height) {
    let water = 0;
	let i=0,
        j = height.length - 1;

	while(i < j){
		let h = Math.min(height[i],height[j]);
		water = Math.max(water,(j-i)*h);

		 if (height[i] < height[j])
                i++;
            else
                j--;
	}

	return water;
};
```

The intuition behind this approach is that the area formed between the lines will always be limited by the height of the shorter line. Further, the farther the lines, the more will be the area obtained.

We take two pointers, one at the beginning and one at the end of the array constituting the length of the lines. Futher, we maintain a variable maxareamaxarea to store the maximum area obtained till now. At every step, we find out the area formed between them, update maxareamaxarea and move the pointer pointing to the shorter line towards the other end by one step.

Initially we consider the area constituting the exterior most lines. Now, to maximize the area, we need to consider the area between the lines of larger lengths. If we try to move the pointer at the longer line inwards, we won't gain any increase in area, since it is limited by the shorter line. But moving the shorter line's pointer could turn out to be beneficial, as per the same argument, despite the reduction in the width. This is done since a relatively longer line obtained by moving the shorter line's pointer might overcome the reduction in area caused by the width reduction.

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
 * @param {number} sum
 * @return {number[][]}
 */
var pathSum = function(root, sum) {
    let paths = getAllPaths(root);

    paths = paths.filter((path) => {
        return path.reduce((a, b) => a + b) === sum;
    })

    return paths;
};

function getAllPaths(root) {

    if(!root) return [];
    let result = [];

    function path(root, arr = []){
        if(!root.left && !root.right) result.push([...arr, root.val]);
        if(root.left) path(root.left, [...arr, root.val]);
        if(root.right) path(root.right, [...arr, root.val]);
    }

    path(root);

    return result;

}
```
We perform a depth first traversal and push in the path when we reach a leaf node / when the root's left and right are null.

## 556. Next Greater Element III

Given a positive 32-bit integer n, you need to find the smallest 32-bit integer which has exactly the same digits existing in the integer n and is greater in value than n. If no such positive 32-bit integer exists, you need to return -1.
Example 1:

Input: 12
Output: 21

Example 2:

Input: 21
Output: -1

```javascript
/**
 * @param {number} n
 * @return {number}
 */
var nextGreaterElement = function(n) {
        number = n.toString().split('');

        let i, j;
        // I) Start from the right most digit and
        // find the first digit that is
        // smaller than the digit next to it.
        for (i = number.length-1; i > 0; i--)
            if (number[i-1] < number[i])
               break;

        // If no such digit is found, its the edge case 1.
        if (i == 0)
            return -1;

         // II) Find the smallest digit on right side of (i-1)'th
         // digit that is greater than number[i-1]
        let x = number[i-1], smallest = i;
        for (j = i+1; j < number.length; j++)
            if (number[j] > x && number[j] <= number[smallest])
                smallest = j;

        // III) Swap the above found smallest digit with
        // number[i-1]
        let temp = number[i-1];
        number[i-1] = number[smallest];
        number[smallest] = temp;

        // IV) Sort the digits after (i-1) in ascending order

        let begin = number.slice(0, i)
        let end = number.slice(i).sort((a, b) => a - b);
        number = begin.concat(end);
        number = parseInt(number.join(''), 10)

        if (number > 2147483647) return -1;

        return number > n ? number : -1;

};
```

## 266. Palindrome Permutation

Given a string, determine if a permutation of the string could form a palindrome.

For example,
`"code"`-> False, `"aab"` -> True, `"carerac"` -> True.

```javascript
var canPermutePalindrom = function(s){
  let frequency = {};

  for (let i = 0; i < s.length; i++){
    frequency[s[i]] ? frequency[s[i]]++ : frequency[s[i]] = 1;
  }

  return Object.values(frequency).filter((count) => count % 2 > 0).length > 1 ? false : true;
}
```

## 110. Balanced Binary Tree

Given a binary tree, determine if it is height-balanced.

For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

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
var isBalanced = function(root) {

    if (root === null)  return true;

    if (Math.abs(getDepth(root.left) - getDepth(root.right)) > 1) return false;

    return isBalanced(root.left) && isBalanced(root.right);
};

function getDepth(root){
    return root === null ? 0 : 1 + Math.max(getDepth(root.left), getDepth(root.right));
}
```

The getDepth function starts to bubble back up when a root is null and 0 is returned to its parent. It is helpful to think in the case of one node to understand the getDepth function. If a root with no children is passed in, then getDepth(root.left) -> getDepth(null) -> returns 0 and the same for the right. Then 0 is returned from both right and left subtrees and 1 is returned from the function because 1+0 = 1.

isBalanced: A tree is balanced if the left and right sub trees are also balanced. We check the height of the left and right subtrees of each node. If the height of all of the subtrees do not differ by more than one than the tree is balanced.

## 208. Implement Trie (Prefix Tree)


Implement a trie with `insert`, `search`, and `startsWith` methods.

Note:
You may assume that all inputs are consist of lowercase letters a-z.

```javascript
function TrieNode() {
    this.children = new Map();
    this.end = false;
}
var Trie = function() {
    this.root = new TrieNode();
};

/**
 * Inserts a word into the trie.
 * @param {string} word
 * @return {void}
 */
Trie.prototype.insert = function(word, node = this.root) {

    if (!word.length) {
        node.end = true;
        return;
    } else if (!node.children.has(word[0])) {
        node.children.set(word[0], new TrieNode())
    }

   return this.insert(word.substr(1), node.children.get(word[0]));

};

/**
 * Returns if the word is in the trie.
 * @param {string} word
 * @return {boolean}
 */
Trie.prototype.search = function(word, node = this.root) {

    if (node.children.has(word[0])) {

        if (node.children.get(word[0]).end && word.length == 1) return true;

        return this.search(word.substr(1), node.children.get(word[0]))
    }

    return false;
};

/**
 * Returns if there is any word in the trie that starts with the given prefix.
 * @param {string} prefix
 * @return {boolean}
 */
Trie.prototype.startsWith = function(prefix, node = this.root) {

    if (!prefix.length) return true;


    if (node.children.has(prefix[0])) {


        return this.startsWith(prefix.substr(1), node.children.get(prefix[0]))
    }

    return false;
};
```

Insert: First we use the root node to insert the first letter of the word. If the root does not have the current letter than we set it as a property. We then recursively call insert on the next letter and pass in the node that we either just set or was already there. We repeat this process until the word length is 0 and then we mark the current node as the end.

Search: We search for a word in a similar fashion. The condition for finding a word is if the current node is the end of the word and if the search term now has a length of 1 (meaning that we have reached the last letter).

##206 Reverse Linked list
do it iteratively and recursively

```javascript
/**
 * Definition for singly-linked list.
 * function ListNode(val) {
 *     this.val = val;
 *     this.next = null;
 * }
 */
/**
 * @param {ListNode} head
 * @return {ListNode}
 */
var reverseList = function(head, prev = null) {

    if (!head) return prev;

    let next = head.next;
    head.next = prev;

    return reverseList(next, head)

};
```

##discuss the differences between python,java and javascript

##1) Given a sorted array of positive and negative integer elements, return an array containing the elements from before but squared and sorted in O(n) time.

[-3, -2, -1, 0, 1, 3, 5] ---> [0, 1, 1, 4, 9, 9, 25]
[-3, -2, 3, 4] ---> [4, 9, 9, 16]

use two pointers. One starts from the end of the negatives and one starts from the start of the positives. insert numbers with absolute value less than the other at the pointer.

```javascript
function squareSorted(arr) {
  let i = 0; //negative pointer
  let j = 0;  //positive pointer

  while(arr[j] < 0) {
    j++;
  }
  i = j - 1;

  let sortedSquare = [];

  while (i >= 0 && j < arr.length) {
    if (Math.abs(arr[i]) < arr[j]) {
      sortedSquare.push(Math.pow(arr[i], 2))
      i--;
    } else {
      sortedSquare.push(Math.pow(arr[j], 2));
      j++
    }
  }

  while (i >= 0) {
    sortedSquare.push(Math.pow(arr[i], 2));
    i--;
  }

  while (j < arr.length) {
    sortedSquare.push(Math.pow(arr[j], 2));
    j++;
  }

  return sortedSquare;
}
```

##lots of people in a running competition and there are lots of check points, build and maintain a real time top ten list.

##Write a function which returns true if is called more than 3 times in 3 seconds.

```javascript
function threeInThree() {
  let counter = 0;

  function called(){
    if (counter > 3) return true;
    counter++;
  }();
}
```

##String Compression

```javascript
var compress = function(chars) {

   let indexAns = 0,
       index = 0;

        while(index < chars.length){
            let currentChar = chars[index];
            let count = 0;
            while(index < chars.length && chars[index] == currentChar){
                index++;
                count++;
            }
            chars[indexAns++] = currentChar;
            if(count != 1)
                for(let c of count.toString().split(''))
                    chars[indexAns++] = c;
        }

        return indexAns;

};
```

##Code to verify if given format is in XML or not
XML documents must have a root element
XML elements must have a closing tag
XML tags are case sensitive
XML elements must be properly nested
XML attribute values must be quoted

##How would you design a music app like Spotify? (Classes and connections between them)

##The difference of thread and process.

A process is a currently running program. A single process or multiple prcesses make up an application. Threads are subsets of processes and A thread is the basic unit for which the computer allocates processing time. A thread can execute any part of the process code, including parts currently being executed by another thread.

For example, a newspaper is like a program. It doesnt have life on its own but it is given life when I read it / when it is executed. When I start to read the arts section that can be a considered a thread of control. When my friend reads the sports section that can also be considered a thread of control. If we both want to then read the business section that is a conflict because there is only one business section and in a computer the operating system needs to solve this conflict. A process is a program plus the state of all threads executing in a program.

##The second question was graphs. Given an undirected graph, find all paths from a target node to source node.
The idea is to do Depth First Traversal of given directed graph. Start the traversal from source. Keep storing the visited vertices in an array say ‘path[]’. If we reach the destination vertex, print contents of path[]. The important thing is to mark current vertices in path[] as visited also, so that the traversal doesn’t go in a cycle.

##Give you an array with increasing order and decreasing order [1,2,3,4,2,1], find a target number.
you can find the maximum (and its position) in O(logn). Then you can just do a binary search in each part which is also O(logn).

```javascript
var search = function(nums, target) {
    let lo = 0, hi = nums.length;

    while (lo < hi) {
        let mid = Math.floor((lo + hi) / 2);

        let num = (nums[mid] < nums[0]) == (target < nums[0]) //are they on the same side
                   ? nums[mid]
                   : target < nums[0] ? -Infinity : Infinity;

        if (num < target)
            lo = mid + 1;
        else if (num > target)
            hi = mid;
        else
            return mid;
    }

    return -1;
};
```

##invert tree
```javascript
var invertTree = function(root) {
     if (root == null) {
            return null;
        }

        let left = root.left,
                right = root.right;

        root.left = invertTree(right);
        root.right = invertTree(left);

        return root;
}
```


##Write a function that finds and returns the node with the second highest value in a Binary Search Tree. Assume the BST is valid, but not necessarily complete or balanced.
inorder traversal

##3 Sum

##Anagrams

##Maximum length of a substring

##If you had n racers and m checkpoints, how would you list out the racers in the order in which they are in the race given that each checkpoint gets a notification when a specific racer crosses it?

##leetcode word break
```javascript
var wordBreak = function(s, wordDict) {

    let queue = [''];
    let memo = new Set();

    while (queue.length){
        let val = queue.shift();


        for (let word of wordDict) {
             let searchWord = `${val}${word}`
             let startsWith = s.indexOf(searchWord) === 0;

            if (s === searchWord) return true;

            if (startsWith && !memo.has(searchWord)) {
                memo.add(searchWord);
                queue.push(searchWord);
            }

        }

    }

    return false;

};
```

##Asked about how to make a binary tree into an array and then how to sort the array with binary search.

##Candy Crush string problem

##sort an array with two stacks.

```javascript
function sort(stack1, stack2) {
  while (stack1.length) {
    let temp = stack1.pop();

    while (stack2.length && temp < stack2[stack2.length - 1]){
      stack1.push(stack2.pop())
    }

    stack2.push(temp)
  }

  return stack2;
}
```

##Implement a schedule that takes as input some number of appointments within a 24 hour period and outputs all the free blocks of time.

##Remove duplicates in a linked list
add seen to a set. If its in seen remove it.

##Find the n-th level of a binary tree,

##Given a string which contains only lowercase letters , remove 3 or more alike letters in sequence so that even after removing there are no 3 or more repeating letters in sequence. Repeat this process as many times as possible.

```javascript

var removeDuplicateLetters = function(s) {

   //creating count array
   const count = new Array(26);
   const a = 'a'.charCodeAt(0);
   for (let i = 0; i < s.length; i++) {
      const k = s.charCodeAt(i) - a;
      count[k] = count[k] ? count[k]+1 : 1;
   }



   const uniqueSubString = [];
   const visited = new Array(26);

  for (let i = 0; i < s.length; i++) {
    const currentLetter = s.charCodeAt(i) - a;
    count[currentLetter] = count[currentLetter] - 1;

    if (visited[currentLetter]) continue;

    while(uniqueSubString.length > 0) {

      const lastLetter = uniqueSubString[uniqueSubString.length-1]-a;

      //if the last letter can come after the currentLetter and there is one of the last letter coming up
      if (lastLetter > currentLetter && count[lastLetter] > 0) {
        visited[lastLetter] = false;
        uniqueSubString.pop();
      } else
        break;
    }

    visited[currentLetter] = true;
    uniqueSubString.push(currentLetter+a);

  }

  return String.fromCharCode(...uniqueSubString);

};
```
